#!/usr/bin/env python3

######################################
# This is a quick scripts to check the logs of make_jsons.py
# and find which remote reads failed
# It supposes you ran make_jsons.py with
# nohup python make_jsons.py > path/to/name.log 2>&1 &
# usage: pyhon find_failed_files.py path/to/name.log

import re
import sys
from collections import defaultdict

def extract_and_group(log_path):
    # match both “Couldn’t process '…'” and “Giving up on file '…'”
    pattern = re.compile(r"""(?:Couldn.?t process|Giving up on file)\s+['\"]([^'\"]+)['\"]""")

    # nested dict: { run_tag: { sample_tag: [ paths... ] } }
    groups = defaultdict(lambda: defaultdict(list))

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            path = m.group(1)
            parts = path.split('/')
            # parts example:
            # ['root:', '', 'cms-xrd-global.cern.ch', '', 'store', 'mc',
            #  'Run3Summer22NanoAODv12',
            #  'DYGto2LG-1Jets_MLL-4to50_PTG-10to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8',
            #  ... ]
            try:
                raw = parts[6]
                sample = parts[7]
            except IndexError:
                raw = 'UNKNOWN'
                sample = 'UNKNOWN'

            # strip "Nano..." from run tag
            run_tag = raw.split('Nano', 1)[0] if 'Nano' in raw else raw

            # strip starting with "_TuneCP5" and replace prefix
            sample_base = sample.split('_TuneCP5', 1)[0]
            sample_tag = sample_base.replace('DYGto2LG-1Jets', 'ZG')

            groups[run_tag][sample_tag].append(path)

    # print grouping
    for run_tag in sorted(groups):
        print(run_tag)
        for sample_tag in sorted(groups[run_tag]):
            print(f"  {sample_tag}")
            for p in groups[run_tag][sample_tag]:
                print(f"    {p}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} PATH/TO/LOGFILE", file=sys.stderr)
        sys.exit(1)
    extract_and_group(sys.argv[1])
