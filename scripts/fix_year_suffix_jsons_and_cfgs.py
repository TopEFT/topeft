#!/usr/bin/env python3
"""
fix_year_suffix_jsons_and_cfgs.py

Goal (canonical convention):
  - For in-scope MC/sample JSONs, filenames must end with _{year-token}.json
  - Year tokens must be suffixes, never prefixes
  - If a filename already contains a year token as prefix or suffix (or both), it is canonicalized to exactly one suffix
  - Directory names that include year tokens are left untouched

Additionally:
  - Rewrite references in:
      * input_samples/cfgs/*.cfg
      * input_samples/sample_jsons/**/*.json  (JSON-to-JSON references)
    so they point to the canonical renamed targets.

CFG path policy (important for your stack):
  - When rewriting cfg references that point under repo_root/input_samples/sample_jsons,
    ALWAYS emit:
        ../../input_samples/sample_jsons/<suffix>
    (even if a shorter ../sample_jsons/... path would work).

Validation:
  - data_samples MUST NOT have year-token suffix
  - in-scope JSONs MUST have canonical suffix
  - cfg references MUST resolve (directly or via legacy-name inference) to an existing JSON
  - JSON references (string tokens that look like .json paths) MUST resolve similarly

IMPORTANT RESOLUTION RULE (fix for ambiguous global search):
  - If a referenced path does not exist, fallback candidates are tried ONLY in the original folder of the reference,
    not anywhere else in sample_jsons.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


YEAR_TOKENS = ["2022", "2022EE", "2023", "2023BPix"]
UL_TOKENS = ["UL16APV", "UL16", "UL17", "UL18"]

RE_YEAR_SUFFIX = re.compile(r"_(2022EE|2023BPix|2022|2023)\.json$", re.IGNORECASE)
RE_YEAR_PREFIX = re.compile(r"^(2022EE|2023BPix|2022|2023)_(.+)\.json$", re.IGNORECASE)

RE_JSON_TOKEN_IN_TEXT = re.compile(r"(?P<path>[^\s\"']+?\.json)")
TRAILING_PUNCT = ",;:)]]}>"
OLD_XROOTD_HOST = "skynet013.crc.nd.edu"
NEW_XROOTD_HOST = "cmsxrootd.crc.nd.edu"

SKIP_SUBPATHS = [
    Path("input_samples/sample_jsons/signal_samples/central_2017"),
    Path("input_samples/sample_jsons/signal_samples/private_TOP19001"),
    Path("input_samples/sample_jsons/signal_samples/private_top19001_local"),
    Path("input_samples/sample_jsons/sync_samples"),
]


def run(cmd: List[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def contains_any_token(s: str, tokens: List[str]) -> bool:
    return any(tok in s for tok in tokens)


def infer_year_token_from_path(p: Path) -> Optional[str]:
    for tok in sorted(YEAR_TOKENS, key=len, reverse=True):
        for seg in p.parts:
            if tok in seg:
                return tok
    return None


def _should_skip_subtree(repo_root: Path, p: Path) -> Tuple[bool, str]:
    pr = p.resolve()
    for sp in SKIP_SUBPATHS:
        sp_abs = (repo_root / sp).resolve()
        if is_under(pr, sp_abs):
            return True, f"skip subtree: {sp.as_posix()}"
    return False, ""


def should_skip_json(repo_root: Path, p: Path, sample_jsons_root: Path) -> Tuple[bool, str]:
    skip, reason = _should_skip_subtree(repo_root, p)
    if skip:
        return True, reason

    rel = p.relative_to(sample_jsons_root).as_posix()

    # NOTE: do not skip data_samples here anymore, we want to enforce rename policy on them.
    # They are out-of-scope for suffix-canonicalization, but they are in-scope for "no suffix" cleanup.
    if contains_any_token(rel, UL_TOKENS):
        return True, "UL token"

    return False, ""


def is_in_scope(repo_root: Path, p: Path, sample_jsons_root: Path) -> Tuple[bool, str, Optional[str]]:
    skip, reason = should_skip_json(repo_root, p, sample_jsons_root)
    if skip:
        return False, reason, None

    rel = p.relative_to(sample_jsons_root).as_posix()

    # data_samples are NOT in-scope for suffix policy; separate policy applies.
    if rel.startswith("data_samples/"):
        return False, "data_samples (special policy: no year-token suffix)", None

    year_tok = infer_year_token_from_path(p)
    if year_tok is None:
        return False, "no year-token inferred (out of scope)", None

    if year_tok not in YEAR_TOKENS:
        return False, f"year-token {year_tok} not in allowed set", None

    return True, "", year_tok


def strip_year_tokens_from_stem(stem: str) -> str:
    changed = True
    out = stem
    while changed:
        changed = False
        for tok in sorted(YEAR_TOKENS, key=len, reverse=True):
            low = out.lower()
            if low.startswith((tok + "_").lower()):
                out = out[len(tok) + 1 :]
                changed = True
                continue
            low = out.lower()
            if low.endswith(("_" + tok).lower()):
                out = out[: -(len(tok) + 1)]
                changed = True
    return out


def canonical_target_for_json(json_path: Path, year_tok: str) -> Path:
    stem = json_path.stem
    base = strip_year_tokens_from_stem(stem)
    new_name = f"{base}_{year_tok}.json"
    return json_path.with_name(new_name)


def canonical_target_for_data_json(json_path: Path) -> Path:
    """
    data_samples policy:
      - MUST NOT have a year-token suffix like _2022.json, _2022EE.json, ...
      - Do NOT try to remove embedded run-era strings like Run2022C (not suffix/prefix token).
    """
    stem = json_path.stem
    base = strip_year_tokens_from_stem(stem)
    return json_path.with_name(f"{base}.json")


def build_rename_plan(repo_root: Path) -> Tuple[Dict[Path, Path], List[str]]:
    sample_jsons_root = repo_root / "input_samples" / "sample_jsons"
    if not sample_jsons_root.is_dir():
        raise RuntimeError(f"Missing folder: {sample_jsons_root}")

    plan: Dict[Path, Path] = {}
    errors: List[str] = []

    all_jsons = sorted(sample_jsons_root.rglob("*.json"))
    for p in all_jsons:
        skip, _ = _should_skip_subtree(repo_root, p)
        if skip:
            continue

        rel = p.relative_to(sample_jsons_root).as_posix()

        # data_samples: enforce "no year-token suffix"
        if rel.startswith("data_samples/"):
            dst = canonical_target_for_data_json(p)
            if dst.resolve() != p.resolve():
                plan[p] = dst
            continue

        in_scope, _, year_tok = is_in_scope(repo_root, p, sample_jsons_root)
        if not in_scope:
            continue
        assert year_tok is not None

        dst = canonical_target_for_json(p, year_tok)
        if dst.resolve() == p.resolve():
            continue
        plan[p] = dst

    inv: Dict[Path, List[Path]] = {}
    for src, dst in plan.items():
        inv.setdefault(dst.resolve(), []).append(src)

    for dst_res, srcs in inv.items():
        if len(srcs) > 1:
            errors.append(
                "Collision: multiple files map to the same target:\n  "
                + "\n  ".join(str(s) for s in srcs)
                + f"\n  -> {dst_res}"
            )

    for src, dst in plan.items():
        if dst.exists() and src.resolve() != dst.resolve():
            errors.append(f"Target already exists: {dst} (from {src})")

    return plan, errors


def git_mv(repo_root: Path, src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, repo_root).replace("\\", "/")
    rel_dst = os.path.relpath(dst, repo_root).replace("\\", "/")

    if dry_run:
        print(f"[DRY-RUN] git mv {rel_src} {rel_dst}")
        return

    run(["git", "mv", rel_src, rel_dst], cwd=repo_root)


def make_abs_map(plan: Dict[Path, Path]) -> Dict[Path, Path]:
    return {src.resolve(): dst.resolve() for src, dst in plan.items()}


def virtual_exists(p: Path, src_to_dst_abs: Dict[Path, Path], dry_run: bool) -> bool:
    pr = p.resolve()
    if not dry_run:
        return pr.exists()

    if pr in src_to_dst_abs:
        return False
    if pr in set(src_to_dst_abs.values()):
        return True
    return pr.exists()


def split_trailing_punct(token: str) -> Tuple[str, str]:
    core = token
    suffix = ""
    while core and core[-1] in set(TRAILING_PUNCT):
        suffix = core[-1] + suffix
        core = core[:-1]
    return core, suffix


def split_comment_prefix(token: str) -> Tuple[str, str]:
    """
    If the token starts with '#' or '# ', strip it for resolution but preserve it for rewriting.

    Examples:
      "#../../a/b.json"  -> ("../../a/b.json", "#")
      "# ../../a/b.json" -> ("../../a/b.json", "# ")
      " ../../a/b.json"  -> (" ../../a/b.json", "")
    """
    if token.startswith("# "):
        return token[2:], "# "
    if token.startswith("#"):
        return token[1:], "#"
    return token, ""


def rewrite_xrootd_host(text: str) -> str:
    return text.replace(OLD_XROOTD_HOST, NEW_XROOTD_HOST)


def remove_year_from_stem(stem: str) -> str:
    return strip_year_tokens_from_stem(stem)


def candidate_names_in_same_folder(original_name: str) -> List[str]:
    stem, ext = os.path.splitext(original_name)
    base = remove_year_from_stem(stem)

    out: List[str] = []
    out.append(base + ext)

    for tok in sorted(YEAR_TOKENS, key=len, reverse=True):
        out.append(f"{base}_{tok}{ext}")

    seen = set()
    dedup: List[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


def mapped_legacy_paths(core: str, base_dir: Path, repo_root: Path) -> List[str]:
    out: List[str] = []

    if core.startswith(("http://", "https://")):
        return out

    norm = core.replace("\\", "/")

    # Legacy: ../../sample_jsons/...  (from input_samples/cfgs this historically pointed at ../sample_jsons/...)
    if "../../sample_jsons/" in norm:
        out.append(norm.replace("../../sample_jsons/", "../sample_jsons/"))

    # Legacy: ../sample_jsons/...  (some stacks want ../../input_samples/sample_jsons/...)
    if "../sample_jsons/" in norm:
        out.append(norm.replace("../sample_jsons/", "../../input_samples/sample_jsons/"))

    # Legacy: ../../input_samples/sample_jsons/...  (sometimes rewritten to ../sample_jsons/...)
    if "../../input_samples/sample_jsons/" in norm:
        out.append(norm.replace("../../input_samples/sample_jsons/", "../sample_jsons/"))

    # Generic mapping: anything containing sample_jsons/<suffix> can be mapped to repo_root/input_samples/sample_jsons/<suffix>
    if "sample_jsons/" in norm:
        idx = norm.find("sample_jsons/")
        suffix = norm[idx + len("sample_jsons/") :]
        target_abs = (repo_root / "input_samples" / "sample_jsons" / suffix).resolve()
        rel_to_base = os.path.relpath(target_abs, base_dir).replace("\\", "/")
        out.append(rel_to_base)

    seen = set()
    dedup: List[str] = []
    for x in out:
        if x not in seen and x != core:
            seen.add(x)
            dedup.append(x)
    return dedup


def resolve_json_reference_scoped(
    token: str,
    base_dir: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
    repo_root: Path,
) -> Tuple[Optional[Path], str]:
    core0, _ = split_trailing_punct(token)
    core, _cprefix = split_comment_prefix(core0)

    if core.startswith(("http://", "https://")):
        return None, "url"

    # Try direct, then mapped legacy variants (still folder-scoped afterwards)
    candidates_core: List[Tuple[str, str]] = [(core, "direct")]
    for alt in mapped_legacy_paths(core, base_dir, repo_root):
        candidates_core.append((alt, "direct-mapped"))

    for core_try, how_tag in candidates_core:
        abs_direct = (base_dir / core_try).resolve()
        if virtual_exists(abs_direct, src_to_dst_abs, dry_run=dry_run):
            return abs_direct, how_tag

        # Folder-scoped fallback candidates (based on the *intended folder*)
        folder = abs_direct.parent
        original_bn = abs_direct.name
        names = candidate_names_in_same_folder(original_bn)

        found: List[Path] = []
        for bn in names:
            cand = (folder / bn).resolve()
            if virtual_exists(cand, src_to_dst_abs, dry_run=dry_run):
                found.append(cand)

        found = list({p.resolve() for p in found})
        if len(found) == 1:
            return found[0], f"folder-fallback ({how_tag})"
        if len(found) > 1:
            basenames = ", ".join(sorted({p.name for p in found}))
            return None, f"ambiguous folder-fallback in {folder}: {basenames}"

    abs_direct0 = (base_dir / core).resolve()
    return None, f"unresolved (scoped to {abs_direct0.parent})"


def format_reference_path(
    target_abs: Path,
    base_dir: Path,
    repo_root: Path,
    *,
    prefer_cfg_input_samples_prefix: bool,
) -> str:
    """
    Format a reference path to `target_abs` from `base_dir`.

    If prefer_cfg_input_samples_prefix is True and target_abs is under:
        repo_root/input_samples/sample_jsons
    then emit the canonical cfg-style path:
        ../../input_samples/sample_jsons/<suffix>

    Otherwise, fall back to os.path.relpath(target_abs, base_dir).
    """
    target_abs = target_abs.resolve()
    sample_root = (repo_root / "input_samples" / "sample_jsons").resolve()

    if prefer_cfg_input_samples_prefix:
        try:
            suffix = target_abs.relative_to(sample_root).as_posix()
            return f"../../input_samples/sample_jsons/{suffix}"
        except Exception:
            pass

    return os.path.relpath(target_abs, base_dir).replace("\\", "/")


def rewrite_json_tokens_in_text(
    text: str,
    base_dir: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
    repo_root: Path,
    *,
    prefer_cfg_input_samples_prefix: bool = False,
) -> Tuple[str, bool, List[str]]:
    changed = False
    notes: List[str] = []

    def repl(m: re.Match) -> str:
        nonlocal changed
        token = m.group("path")
        core_with_punct, punct = split_trailing_punct(token)
        core, cprefix = split_comment_prefix(core_with_punct)

        abs_res, how = resolve_json_reference_scoped(
            core,
            base_dir,
            src_to_dst_abs,
            dry_run=dry_run,
            repo_root=repo_root,
        )

        if abs_res is None:
            if how not in ("url",):
                notes.append(f"unresolved token '{core}' ({how}) in {base_dir}")
            return token

        abs_res_r = abs_res.resolve()
        abs_dst = src_to_dst_abs.get(abs_res_r)
        target = abs_dst if abs_dst is not None else abs_res_r

        new_token = format_reference_path(
            target,
            base_dir,
            repo_root,
            prefer_cfg_input_samples_prefix=prefer_cfg_input_samples_prefix,
        )
        out = cprefix + new_token + punct
        if out != token:
            changed = True
        return out

    new_text = RE_JSON_TOKEN_IN_TEXT.sub(repl, text)
    return new_text, changed, notes


def rewrite_cfg_line(
    line: str,
    base_dir: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
    repo_root: Path,
    delete_unresolved: bool,
) -> Tuple[Optional[str], bool, List[str]]:
    """
    Rewrites all .json tokens in a single cfg line.
    - Processes commented lines too (leading '#', '# ' preserved via split_comment_prefix()).
    - If delete_unresolved is True: drop the whole line if ANY non-url .json token is unresolved
      (except tokens under topcoffea/json/, which are treated as out-of-scope).
    Returns:
      (new_line_or_None_if_dropped, changed, notes)
    """
    changed = False
    notes: List[str] = []
    had_unresolved = False

    def repl(m: re.Match) -> str:
        nonlocal changed, had_unresolved
        token = m.group("path")
        core_with_punct, punct = split_trailing_punct(token)
        core, cprefix = split_comment_prefix(core_with_punct)

        # out-of-scope: don't enforce, don't drop
        if "topcoffea/json/" in core.replace("\\", "/"):
            return token

        abs_res, how = resolve_json_reference_scoped(
            core,
            base_dir,
            src_to_dst_abs,
            dry_run=dry_run,
            repo_root=repo_root,
        )

        if abs_res is None:
            if how != "url":
                had_unresolved = True
                notes.append(f"unresolved token '{core}' ({how}) in {base_dir}")
            return token

        abs_res_r = abs_res.resolve()
        abs_dst = src_to_dst_abs.get(abs_res_r)
        target = abs_dst if abs_dst is not None else abs_res_r

        new_token = format_reference_path(
            target,
            base_dir,
            repo_root,
            prefer_cfg_input_samples_prefix=True,
        )
        out = cprefix + new_token + punct
        if out != token:
            changed = True
        return out

    new_line = RE_JSON_TOKEN_IN_TEXT.sub(repl, line)

    if delete_unresolved and had_unresolved:
        return None, True, notes  # treat as changed since we remove the line

    return new_line, changed, notes


def update_cfgs(
    repo_root: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
    delete_unresolved_cfg_lines: bool,
) -> int:
    cfg_root = repo_root / "input_samples" / "cfgs"
    if not cfg_root.is_dir():
        print(f"Warning: cfg folder not found: {cfg_root}")
        return 0

    changed_files = 0
    for cfg in sorted(cfg_root.rglob("*.cfg")):
        original = cfg.read_text()
        out_lines: List[str] = []
        changed = False

        for line in original.splitlines(keepends=True):
            rewritten_line = rewrite_xrootd_host(line)
            host_changed = rewritten_line != line
            if host_changed:
                changed = True

            stripped = rewritten_line.strip()
            if not stripped:
                out_lines.append(rewritten_line)
                continue

            # Keep your existing "skip directory markers" behavior
            if stripped.endswith("/"):
                out_lines.append(rewritten_line)
                continue

            new_line, line_changed, _notes = rewrite_cfg_line(
                line=rewritten_line,
                base_dir=cfg.parent,
                src_to_dst_abs=src_to_dst_abs,
                dry_run=dry_run,
                repo_root=repo_root,
                delete_unresolved=delete_unresolved_cfg_lines,
            )

            if new_line is None:
                changed = True
                continue  # line dropped

            if line_changed:
                changed = True
            out_lines.append(new_line)

        if changed:
            changed_files += 1
            if dry_run:
                print(f"[DRY-RUN] would update cfg: {cfg}")
            else:
                cfg.write_text("".join(out_lines))

    return changed_files


def _walk_json(obj: Any) -> Iterable[Tuple[List[Union[str, int]], Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            for subp, subv in _walk_json(v):
                yield [k] + subp, subv
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            for subp, subv in _walk_json(v):
                yield [i] + subp, subv
    else:
        yield [], obj


def _set_json_value(root: Any, path: List[Union[str, int]], value: Any) -> None:
    cur = root
    for p in path[:-1]:
        cur = cur[p]
    if path:
        cur[path[-1]] = value


def update_json_references(
    repo_root: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
) -> int:
    sample_jsons_root = repo_root / "input_samples" / "sample_jsons"
    if not sample_jsons_root.is_dir():
        return 0

    changed_files = 0
    for jp in sorted(sample_jsons_root.rglob("*.json")):
        skip, _ = _should_skip_subtree(repo_root, jp)
        if skip:
            continue

        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue

        changed = False
        for path, value in _walk_json(data):
            if not isinstance(value, str):
                continue

            new_val = rewrite_xrootd_host(value)

            if ".json" in new_val:
                rewritten_json_refs, _val_changed, _notes = rewrite_json_tokens_in_text(
                    new_val,
                    jp.parent,
                    src_to_dst_abs,
                    dry_run=dry_run,
                    repo_root=repo_root,
                    prefer_cfg_input_samples_prefix=False,  # keep JSON-to-JSON refs naturally relative
                )
                new_val = rewritten_json_refs

            if new_val != value:
                _set_json_value(data, path, new_val)
                changed = True

        if changed:
            changed_files += 1
            if dry_run:
                print(f"[DRY-RUN] would update json refs: {jp}")
            else:
                jp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    return changed_files


def validate(
    repo_root: Path,
    src_to_dst_abs: Dict[Path, Path],
    dry_run: bool,
) -> int:
    problems = 0
    sample_jsons_root = repo_root / "input_samples" / "sample_jsons"

    # 1) Enforce: data_samples MUST NOT have year-token suffix in filename
    data_root = sample_jsons_root / "data_samples"
    if data_root.is_dir():
        for p in sorted(data_root.rglob("*.json")):
            if RE_YEAR_SUFFIX.search(p.name):
                print(f"[PROBLEM] data_samples json has year-token suffix: {p}")
                problems += 1

    # 2) Enforce: in-scope JSONs must have canonical suffix-only naming
    if sample_jsons_root.is_dir():
        for p in sorted(sample_jsons_root.rglob("*.json")):
            in_scope, _, year_tok = is_in_scope(repo_root, p, sample_jsons_root)
            if not in_scope:
                continue
            assert year_tok is not None
            expected = canonical_target_for_json(p, year_tok)
            if p.name != expected.name:
                print(
                    "[PROBLEM] non-canonical filename (want suffix-only canonical): "
                    f"{p}  (expected name: {expected.name})"
                )
                problems += 1

    # 3) Validate cfg references (folder-scoped resolution + legacy path mapping)
    cfg_root = repo_root / "input_samples" / "cfgs"
    if cfg_root.is_dir():
        for cfg in sorted(cfg_root.rglob("*.cfg")):
            for raw_line in cfg.read_text().splitlines():
                raw = raw_line.strip()
                if not raw or raw.endswith("/"):
                    continue

                for m in RE_JSON_TOKEN_IN_TEXT.finditer(raw_line):
                    token = m.group("path")
                    core_with_punct, _punct = split_trailing_punct(token)
                    core, _cprefix = split_comment_prefix(core_with_punct)

                    # Out-of-scope: do not enforce existence for topcoffea json trees (UL/private/debug)
                    if "topcoffea/json/" in core.replace("\\", "/"):
                        continue

                    abs_res, how = resolve_json_reference_scoped(
                        core,
                        cfg.parent,
                        src_to_dst_abs,
                        dry_run=dry_run,
                        repo_root=repo_root,
                    )
                    if abs_res is None and how != "url":
                        print(f"[PROBLEM] cfg json reference unresolved: {cfg}:{core} ({how})")
                        problems += 1

    # 4) Validate JSON-to-JSON references under sample_jsons (same resolution rules)
    if sample_jsons_root.is_dir():
        for jp in sorted(sample_jsons_root.rglob("*.json")):
            skip, _ = _should_skip_subtree(repo_root, jp)
            if skip:
                continue
            try:
                data = json.loads(jp.read_text())
            except Exception:
                continue

            for _, value in _walk_json(data):
                if not isinstance(value, str) or ".json" not in value:
                    continue

                for m in RE_JSON_TOKEN_IN_TEXT.finditer(value):
                    token = m.group("path")
                    core_with_punct, _punct = split_trailing_punct(token)
                    core, _cprefix = split_comment_prefix(core_with_punct)

                    if "topcoffea/json/" in core.replace("\\", "/"):
                        continue

                    abs_res, how = resolve_json_reference_scoped(
                        core,
                        jp.parent,
                        src_to_dst_abs,
                        dry_run=dry_run,
                        repo_root=repo_root,
                    )
                    if abs_res is None and how != "url":
                        print(f"[PROBLEM] json reference unresolved: {jp}:{core} ({how})")
                        problems += 1

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Path to repo root")

    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    mx.add_argument("--dry-run", action="store_true", help="Dry-run (prints planned actions)")

    ap.add_argument("--no-cfg", action="store_true", help="Do not rewrite cfg files")
    ap.add_argument("--no-json-refs", action="store_true", help="Do not rewrite JSON-to-JSON references")
    ap.add_argument("--validate-only", action="store_true", help="Run validations only, no rename/rewrite")

    ap.add_argument(
        "--delete-unresolved-cfg-lines",
        action="store_true",
        help="When rewriting cfgs, drop any line (including commented ones) that contains an unresolved .json token",
    )

    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()

    dry_run = True
    if args.apply:
        dry_run = False
    elif args.dry_run:
        dry_run = True

    plan, errors = build_rename_plan(repo_root)
    if errors:
        print("Errors detected (rename plan):")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(2)

    src_to_dst_abs = make_abs_map(plan)

    if args.validate_only:
        problems = validate(repo_root, src_to_dst_abs, dry_run=dry_run)
        if problems:
            raise SystemExit(f"Validation failed with {problems} problem(s).")
        print("Validation OK.")
        return

    if not plan:
        print("Nothing to rename.")
    else:
        print(f"Planned renames (canonical policy incl. data_samples cleanup): {len(plan)}")
        for src, dst in plan.items():
            rel_src = os.path.relpath(src, repo_root).replace("\\", "/")
            rel_dst = os.path.relpath(dst, repo_root).replace("\\", "/")
            print(f"  {rel_src}  ->  {rel_dst}")

        for src, dst in plan.items():
            git_mv(repo_root, src, dst, dry_run=dry_run)

    if not args.no_cfg:
        changed = update_cfgs(
            repo_root,
            src_to_dst_abs,
            dry_run=dry_run,
            delete_unresolved_cfg_lines=args.delete_unresolved_cfg_lines,
        )
        print(f"cfg files updated: {changed}")

    if not args.no_json_refs:
        changed = update_json_references(repo_root, src_to_dst_abs, dry_run=dry_run)
        print(f"json reference files updated: {changed}")

    problems = validate(repo_root, src_to_dst_abs, dry_run=dry_run)
    if problems:
        raise SystemExit(f"Post-change validation failed with {problems} problem(s).")
    print("Post-change validation OK.")


if __name__ == "__main__":
    main()
