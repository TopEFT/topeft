import os
import shutil
import argparse
import json
import sys
from typing import Iterable, List, Tuple

import yaml

from analysis.topeft_run2.scenario_registry import resolve_scenario_choice
from topeft.modules.channel_metadata import ChannelMetadataHelper

# This script does some basic checks of the cards and templates produced by the `make_cards.py` script.
#   - It also can parse the condor log files and dump a summary of the contents
#   - Additionally, it can also grab the right set of ptz and lj0pt templates (for the right categories) used in TOP-22-006

# Lines that show up in the condor err files that we want to ignore
IGNORE_LINES = [
    "FutureWarning: In coffea version v2023.3.0 (target date: 31 Mar 2023), this will be an error.",
    "(Set coffea.deprecations_as_errors = True to get a stack trace now.)",
    "ImportError: coffea.hist is deprecated",
    "warnings.warn(message, FutureWarning)",
]

# Default scenario to mirror run_analysis.py
DEFAULT_SCENARIO = "TOP_22_006"

# Historical expected file counts for standard scenarios (text, root).
# Acts as a lightweight safety net when copying templates.
EXPECTED_FILE_COUNTS = {
    "TOP_22_006": (43, 43),
}


# Return list of lines in a file
def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

# Check if we want to ignore the line or not (based on whether or not any of a list of strings we don't care about shows up in the line)
def ignore_line(line_to_check,list_of_str_to_ignore=IGNORE_LINES):
    ignore = False
    for str_to_ignore in list_of_str_to_ignore:
        if str_to_ignore in line_to_check:
            ignore = True
    return ignore

def extract_number(item):
    return str(''.join(char for char in item if char.isdigit()))


def determine_histogram_suffix(region, jet_value, analysis_mode):
    """Infer the histogram name suffix for a region based on metadata tags."""

    tags = set(region.tags)
    subchannel = region.subchannel or ""
    channel = region.channel or ""
    name = region.name
    jet_value = str(jet_value) if jet_value is not None else ""

    if "onZ_tau" in tags:
        return "ptz_wtau"

    if analysis_mode == "fwd":
        if channel and "2lss_fwd" in channel:
            return "lt"
        if "~fwdjet_mask" in tags and channel == "2lss":
            return "lt"

    if subchannel == "3l_onZ" or channel == "3l_onZ":
        if name == "3l_onZ_2b" and jet_value not in {"4", "5"}:
            return "lj0pt"
        return "ptz"

    if analysis_mode == "offZdivision" and (
        "offZ_low" in subchannel or "offZ_high" in subchannel
    ):
        return "ptz"

    if "offZ_tau" in tags:
        return "ptz"

    if "2l_onZ" in tags or "2l_onZ_as" in tags:
        return "ptz"

    if analysis_mode == "tau" and channel == "2los":
        return "ptz"

    return "lj0pt"

def _analysis_mode_for_group(group, scenario_name):
    """Return the legacy analysis-mode tag for ``group``.

    The analysis mode feeds into ``determine_histogram_suffix`` to preserve the
    historical `_ptz`, `_lj0pt`, `_lt`, and `_ptz_wtau` suffixes that datacards
    expect.  Features recorded on the channel group (or the high-level scenario)
    drive the mapping:

    * ``offz_split`` → ``"offZdivision"``
    * ``requires_tau`` or tau scenario → ``"tau"``
    * ``requires_forward`` or forward scenario → ``"fwd"``
    * otherwise → ``"top22006"`` (baseline behaviour)
    """

    features = set(group.features)
    if "offz_split" in features:
        return "offZdivision"
    if "requires_tau" in features or scenario_name == "tau_analysis":
        return "tau"
    if "requires_forward" in features or scenario_name == "fwd_analysis":
        return "fwd"
    return "top22006"


def resolve_scenario_metadata(scenario_args: Iterable[str]) -> Tuple[str, str, ChannelMetadataHelper]:
    """Return the scenario name, metadata path, and helper for ``scenario_args``."""

    scenario_names = [name for name in (scenario_args or []) if name]
    if not scenario_names:
        scenario_names = [DEFAULT_SCENARIO]

    if len(scenario_names) != 1:
        raise ValueError(
            "Datacard tooling currently supports one scenario per run. "
            f"Requested scenarios: {', '.join(scenario_names)}"
        )

    scenario_name = scenario_names[0]
    resolution = resolve_scenario_choice(scenario_name)

    with open(resolution.metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = yaml.safe_load(metadata_file) or {}

    channels_metadata = metadata.get("channels")
    if not channels_metadata:
        raise ValueError(
            f"Channel metadata is missing from the metadata YAML ({resolution.metadata_path})."
        )

    helper = ChannelMetadataHelper(channels_metadata)
    return scenario_name, resolution.metadata_path, helper


def collect_datacard_channels(
    channel_helper: ChannelMetadataHelper, scenario_name: str
) -> List[str]:
    """Return the canonical datacard channel list for ``scenario_name``.

    This helper is the single source of truth for channel naming and is shared
    by both ``datacards_post_processing.py`` and ``make_cards.py`` to guarantee
    that template copying, WC selection, and condor jobs stay in sync.
    """

    channel_names: List[str] = []
    group_names = channel_helper.selected_group_names([scenario_name])
    for group_name in group_names:
        if group_name.endswith("_CR"):
            continue
        group = channel_helper.group(group_name)
        analysis_mode = _analysis_mode_for_group(group, scenario_name)
        for category in group.categories():
            jet_bins = [extract_number(item) for item in category.jet_bins]
            jet_bins = [jet for jet in jet_bins if jet]
            for region in category.region_definitions:
                for jet in jet_bins:
                    hist_suffix = determine_histogram_suffix(region, jet, analysis_mode)
                    channel_names.append(f"{region.name}_{jet}j_{hist_suffix}")

    seen = set()
    ordered_unique: List[str] = []
    for name in sorted(channel_names):
        if name in seen:
            continue
        seen.add(name)
        ordered_unique.append(name)
    return ordered_unique


# Check the output of the datacard maker
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("datacards_path", help="The path to the directory with the datacards in it.")
    parser.add_argument(
        "-c",
        "--check-condor-logs",
        action="store_true",
        help="Check the contents of the condor err files.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help=(
            "Scenario name to copy channels for (e.g. TOP_22_006, tau_analysis, fwd_analysis). "
            "Only one scenario per run is currently supported."
        ),
    )
    args = parser.parse_args()

    try:
        scenario_name, metadata_path, channel_helper = resolve_scenario_metadata(args.scenario)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    CATSELECTED = collect_datacard_channels(channel_helper, scenario_name)
    if not CATSELECTED:
        raise ValueError(f"No signal-region channels found for scenario '{scenario_name}'.")

    ###### Print out general info ######

    with open(os.path.join(args.datacards_path,'scalings-preselect.json'), 'r') as file:
        scalings_content = json.load(file)

    # Count the number of text data cards and root templates
    n_text_cards = 0
    n_root_templates = 0
    datacard_files = os.listdir(args.datacards_path)
    for fname in datacard_files:
        if fname.startswith("ttx_multileptons") and fname.endswith(".txt"):
            n_text_cards += 1
        if fname.startswith("ttx_multileptons") and fname.endswith(".root"):
            n_root_templates += 1

    # Print out what we learned
    print(f"\nSummary of cards and templates in {args.datacards_path}:")
    print(f"\tNumber of text cards    : {n_text_cards}")
    print(f"\tNumber of root templates: {n_root_templates}")


    # Check the condor err files
    if args.check_condor_logs:
        lines_from_condor_err_to_print = []
        lines_from_condor_out_to_print = []
        condor_logs_path = os.path.join(args.datacards_path,"job_logs")
        condor_log_files = os.listdir(condor_logs_path)
        for fname in condor_log_files:
            # Parse the .err files
            if fname.endswith(".err"):
                err_file_lines = read_file(os.path.join(condor_logs_path,fname))
                for line in err_file_lines:
                    if not ignore_line(line):
                        lines_from_condor_err_to_print.append((fname,line))
            # Parse the .out files
            if fname.endswith(".out"):
                out_file_lines = read_file(os.path.join(condor_logs_path,fname))
                for line in out_file_lines:
                    if "ERROR" in line:
                        lines_from_condor_out_to_print.append((fname,line))

        # Print out what we learned
        print(f"\nSummary of condor err files in {condor_logs_path}:")
        print(f"\tNumber of non-ingnored lines in condor err files: {len(lines_from_condor_err_to_print)}")
        for line in lines_from_condor_err_to_print:
            print(f"\t\t* In {line[0]}: {line[1]}")
        print(f"\tNumber of ERROR lines in condor out files: {len(lines_from_condor_out_to_print)}")
        for line in lines_from_condor_out_to_print:
            print(f"\t\t* In {line[0]}: {line[1]}")


    # Grab the ptz/lj0pt/lt cards we want for the selected scenario
    n_txt = 0
    n_root = 0
    ptzlj0pt_path = os.path.join(args.datacards_path,"ptz-lj0pt_withSys")
    os.mkdir(ptzlj0pt_path)
    print(f"\nCopying {scenario_name} relevant files to {ptzlj0pt_path}...")

    for fname in datacard_files:
        file_name_strip_ext = os.path.splitext(fname)[0]
        for file in CATSELECTED:
            if file in file_name_strip_ext:
                shutil.copyfile(os.path.join(args.datacards_path,fname),os.path.join(ptzlj0pt_path,fname))
                if fname.endswith(".txt"): n_txt += 1
                if fname.endswith(".root"): n_root += 1
    #also copy the selectedWCs.txt file
    shutil.copyfile(os.path.join(args.datacards_path,"selectedWCs.txt"),os.path.join(ptzlj0pt_path,"selectedWCs.txt"))

    for item in scalings_content:
        channel_name = item.get("channel")
        if channel_name in CATSELECTED:
            ch_index = CATSELECTED.index(channel_name) + 1
            item["channel"] = "ch" + str(ch_index)
        else:
            scalings_content = [d for d in scalings_content if d != item]

    with open(os.path.join(ptzlj0pt_path, 'scalings.json'), 'w') as file:
        json.dump(scalings_content, file, indent=4)

    # Check that we got the expected number and print what we learn
    print(f"\tNumber of text templates copied: {n_txt}")
    print(f"\tNumber of root templates copied: {n_root}")

    expected_counts = EXPECTED_FILE_COUNTS.get(scenario_name)
    if expected_counts is not None:
        exp_txt, exp_root = expected_counts
        if n_txt != exp_txt or n_root != exp_root:
            raise Exception(
                "Unexpected number of files copied for scenario "
                f"'{scenario_name}'. Expected {exp_txt} text and {exp_root} root "
                f"templates, saw {n_txt} text and {n_root} root."
            )
    else:
        print(
            f"\tNo reference file counts registered for scenario '{scenario_name}'; "
            "skipping sanity check."
        )

    print("Done.\n")


if __name__ == "__main__":
    main()
