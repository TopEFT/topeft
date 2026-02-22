from __future__ import annotations

import json

import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


def test_parser_accepts_repeated_f_and_defaults_collision_policy_error():
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args(["-f", "a.pkl.gz", "-f", "b.pkl.gz"])

    assert args.pkl_file_path == ["a.pkl.gz", "b.pkl.gz"]
    assert args.on_process_collision == "error"


def test_resolve_pkl_paths_from_repeated_f_only():
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args(["-f", "a.pkl.gz", "-f", "b.pkl.gz"])

    resolved = make_cr_and_sr_plots._resolve_pkl_paths(args, parser)

    assert resolved == ["a.pkl.gz", "b.pkl.gz"]


def test_resolve_pkl_paths_from_list_file_only(tmp_path):
    pkl_list = tmp_path / "pkls.txt"
    pkl_list.write_text("# comment\n\n/tmp/a.pkl.gz\n/tmp/b.pkl.gz\n")

    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args(["--pkl-list-file", str(pkl_list)])

    resolved = make_cr_and_sr_plots._resolve_pkl_paths(args, parser)

    assert resolved == ["/tmp/a.pkl.gz", "/tmp/b.pkl.gz"]


def test_resolve_pkl_paths_errors_when_mixing_repeated_f_and_list_file(tmp_path):
    pkl_list = tmp_path / "pkls.txt"
    pkl_list.write_text("/tmp/a.pkl.gz\n")

    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args(["-f", "one.pkl.gz", "--pkl-list-file", str(pkl_list)])

    with pytest.raises(SystemExit):
        make_cr_and_sr_plots._resolve_pkl_paths(args, parser)


def test_merge_only_short_circuits_before_plotting(monkeypatch, tmp_path):
    parser = make_cr_and_sr_plots.build_arg_parser()
    args = parser.parse_args(
        [
            "-f",
            "input.pkl.gz",
            "-o",
            str(tmp_path),
            "-n",
            "out",
            "--merge-only",
            "--merge-report",
            "merge_report.json",
        ]
    )

    fake_report = {
        "num_inputs": 1,
        "num_merged_keys": 1,
        "num_process_collisions": 0,
        "process_collisions": [],
    }

    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "load_and_merge_histogram_pkls",
        lambda *_, **__: ({"met": object()}, fake_report),
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_plots_for_region should not be called in --merge-only mode")

    monkeypatch.setattr(make_cr_and_sr_plots, "run_plots_for_region", _fail_if_called)

    exit_code = make_cr_and_sr_plots.run_with_args(args, parser)

    assert exit_code == 0
    report_path = tmp_path / "out" / "merge_report.json"
    assert report_path.exists()
    report_data = json.loads(report_path.read_text())
    assert report_data["num_inputs"] == 1
    assert report_data["num_process_collisions"] == 0
