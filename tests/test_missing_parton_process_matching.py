from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import uproot


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "topeft_run2"
    / "missing_parton.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "missing_parton_process_matching_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_card(tmp_path, *, root_processes, txt_processes):
    root_path = tmp_path / "ttx_multileptons-2lss_p_njets.root"
    txt_path = root_path.with_suffix(".txt")
    values = np.arange(1.0, 9.0)
    edges = np.arange(9.0)
    with uproot.recreate(root_path) as root_file:
        for process in root_processes:
            root_file[f"{process}_sm"] = (values, edges)
    txt_path.write_text(
        "process " + " ".join(f"{process}_sm" for process in txt_processes) + "\n"
        "process " + " ".join(str(index) for index in range(len(txt_processes))) + "\n"
        "rate " + " ".join("36.0" for _ in txt_processes) + "\n",
        encoding="utf-8",
    )
    return root_path, txt_path


@pytest.mark.parametrize(
    "root_key",
    (
        "tZq_sm;1",
        "TZQB-Zto2L-4FS_MLL-30_sm;1",
        " TZQB-Zto2L-4FS_MLL-30_sm;1",
    ),
)
def test_every_documented_central_alias_matches_exactly(root_key):
    module = load_module()

    assert module.is_nominal_key(root_key, module.DEFAULT_CENTRAL_PROCESS)


@pytest.mark.parametrize("root_key", ("tllq_sm;1", " tllq_sm;1"))
def test_every_documented_private_alias_matches_exactly(root_key):
    module = load_module()

    assert module.is_nominal_key(root_key, module.DEFAULT_PRIVATE_PROCESS)


def test_only_leading_and_trailing_whitespace_are_normalized():
    module = load_module()

    assert module.matches_process_name(" tllq_sm ", "tllq")
    assert not module.matches_process_name("t llq_sm", "tllq")
    assert not module.matches_process_name("prefix_tllq_sm", "tllq")


def test_tzq_is_never_classified_as_private_tllq():
    module = load_module()

    for central_name in module.PROCESS_ALIASES[module.DEFAULT_CENTRAL_PROCESS]:
        assert not module.matches_process_name(
            f"{central_name}_sm",
            module.DEFAULT_PRIVATE_PROCESS,
        )


def test_nonprompt_is_never_classified_as_signal():
    module = load_module()

    assert not module.matches_process_name("nonprompt_sm", "tllq")
    assert not module.matches_process_name("nonprompt_sm", "tZq")


def test_substring_collisions_are_rejected():
    module = load_module()

    assert not module.matches_process_name("my_tllq_background_sm", "tllq")
    assert not module.matches_process_name(
        "TZQB-Zto2L-4FS_MLL-30_extra_sm",
        "tZq",
    )


def test_zero_root_match_reports_all_available_processes(tmp_path):
    module = load_module()
    root_path, txt_path = write_card(
        tmp_path,
        root_processes=("nonprompt",),
        txt_processes=("tllq",),
    )

    with pytest.raises(ValueError, match="exactly one nominal ROOT") as exc_info:
        module.read_base_category_card(
            module.CardFiles(root_path, txt_path),
            "tllq",
            base_channel="2lss_p",
            role="private",
        )

    assert "available_processes=('nonprompt',)" in str(exc_info.value)


def test_multiple_semantic_root_matches_fail(tmp_path):
    module = load_module()
    root_path, txt_path = write_card(
        tmp_path,
        root_processes=("tZq", "TZQB-Zto2L-4FS_MLL-30"),
        txt_processes=("tZq",),
    )

    with pytest.raises(ValueError, match="exactly one nominal ROOT") as exc_info:
        module.read_base_category_card(
            module.CardFiles(root_path, txt_path),
            "tZq",
            base_channel="2lss_p",
            role="central",
        )

    message = str(exc_info.value)
    assert "tZq_sm" in message
    assert "TZQB-Zto2L-4FS_MLL-30_sm" in message


def test_multiple_semantic_txt_matches_fail(tmp_path):
    module = load_module()
    root_path, txt_path = write_card(
        tmp_path,
        root_processes=("tZq",),
        txt_processes=("tZq", "TZQB-Zto2L-4FS_MLL-30"),
    )

    with pytest.raises(ValueError, match="exactly one TXT semantic process"):
        module.read_base_category_card(
            module.CardFiles(root_path, txt_path),
            "tZq",
            base_channel="2lss_p",
            role="central",
        )
