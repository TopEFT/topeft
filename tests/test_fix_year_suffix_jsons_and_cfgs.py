import importlib.util
import json
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/fix_year_suffix_jsons_and_cfgs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fix_year_suffix_jsons_and_cfgs_test", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _touch_json(path: Path) -> None:
    _write_json(path, {})


def _make_repo_layout(tmp_path: Path):
    repo_root = tmp_path / "repo"
    cfg_dir = repo_root / "input_samples" / "cfgs"
    sample_root = repo_root / "input_samples" / "sample_jsons"
    cfg_dir.mkdir(parents=True)
    sample_root.mkdir(parents=True)
    return repo_root, cfg_dir, sample_root


def test_rewrite_cfg_line_canonical_paths_and_comment_prefixes(tmp_path):
    module = _load_module()
    repo_root, cfg_dir, sample_root = _make_repo_layout(tmp_path)

    _touch_json(sample_root / "signal_samples/ND_skim2022EE/tttt_2022EE.json")
    _touch_json(sample_root / "background_samples/ND_2022EE/TTG-1Jets_PTG-200_2022EE.json")
    _touch_json(sample_root / "background_samples/ND_2022EE/TTto2L2Nu-2Jets_2022EE.json")

    cases = [
        (
            "../sample_jsons/signal_samples/ND_skim2022EE/tttt_2022EE.json\n",
            "../../input_samples/sample_jsons/signal_samples/ND_skim2022EE/tttt_2022EE.json\n",
        ),
        (
            "# ../sample_jsons/background_samples/ND_2022EE/TTG-1Jets_PTG-200_2022EE.json\n",
            "# ../../input_samples/sample_jsons/background_samples/ND_2022EE/TTG-1Jets_PTG-200_2022EE.json\n",
        ),
        (
            "# #../sample_jsons/background_samples/ND_2022EE/TTto2L2Nu-2Jets_2022EE.json\n",
            "# #../../input_samples/sample_jsons/background_samples/ND_2022EE/TTto2L2Nu-2Jets_2022EE.json\n",
        ),
    ]

    for line, expected in cases:
        new_line, changed, notes = module.rewrite_cfg_line(
            line=line,
            base_dir=cfg_dir,
            src_to_dst_abs={},
            dry_run=False,
            repo_root=repo_root,
            delete_unresolved=False,
        )
        assert new_line == expected
        assert changed
        assert notes == []


def test_update_cfgs_rewrites_xrootd_host_and_keeps_canonical_cfg_paths(tmp_path):
    module = _load_module()
    repo_root, cfg_dir, sample_root = _make_repo_layout(tmp_path)

    _touch_json(sample_root / "signal_samples/ND_skim2022EE/tttt_2022EE.json")

    cfg_path = cfg_dir / "test.cfg"
    cfg_path.write_text(
        "root://skynet013.crc.nd.edu//store/mc/\n"
        "# root://skynet013.crc.nd.edu//store/data/\n"
        "../sample_jsons/signal_samples/ND_skim2022EE/tttt_2022EE.json\n",
        encoding="utf-8",
    )

    changed_files = module.update_cfgs(
        repo_root=repo_root,
        src_to_dst_abs={},
        dry_run=False,
        delete_unresolved_cfg_lines=False,
    )

    assert changed_files == 1

    updated = cfg_path.read_text(encoding="utf-8")
    assert "skynet013.crc.nd.edu" not in updated
    assert "root://cmsxrootd.crc.nd.edu//store/mc/" in updated
    assert "# root://cmsxrootd.crc.nd.edu//store/data/" in updated
    assert (
        "../../input_samples/sample_jsons/signal_samples/ND_skim2022EE/tttt_2022EE.json"
        in updated
    )


def test_update_json_references_rewrites_xrootd_host_in_string_values(tmp_path):
    module = _load_module()
    repo_root, _cfg_dir, sample_root = _make_repo_layout(tmp_path)

    json_path = sample_root / "example/a.json"
    _touch_json(sample_root / "example/b.json")
    _write_json(
        json_path,
        {
            "redirector": "root://skynet013.crc.nd.edu//store/test.root",
            "refs": ["./b.json", "root://skynet013.crc.nd.edu//store/another.root"],
        },
    )

    changed_files = module.update_json_references(
        repo_root=repo_root,
        src_to_dst_abs={},
        dry_run=False,
    )
    assert changed_files == 1

    updated = json.loads(json_path.read_text(encoding="utf-8"))
    assert updated["redirector"] == "root://cmsxrootd.crc.nd.edu//store/test.root"
    assert updated["refs"][0] == "b.json"
    assert updated["refs"][1] == "root://cmsxrootd.crc.nd.edu//store/another.root"
