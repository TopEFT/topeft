#!/usr/bin/env python
"""Run the tutorial processor and emit tuple-keyed histogram summaries.

The emitted pickle contains an :class:`collections.OrderedDict` keyed by
``(variable, channel, application, sample, systematic)`` tuples so that
downstream training code can deterministically access histogram contents.
"""

import json
import time
import cloudpickle
import gzip
import os
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import hist
import coffea.processor as processor
from coffea.nanoevents import NanoAODSchema

from topcoffea.modules.histEFT import HistEFT

import simple_processor


def _ensure_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    return np.asarray(array)


def _summarise_hist(histogram: Any) -> Dict[str, Any]:
    """Create a tuple-aware summary for a histogram instance."""

    values: Optional[np.ndarray] = None
    variances: Optional[np.ndarray] = None

    if isinstance(histogram, HistEFT):
        entries = histogram.values(sumw2=True)
        for payload in entries.values():
            if isinstance(payload, tuple):
                current_values = _ensure_numpy(payload[0])
                current_variances = None if payload[1] is None else _ensure_numpy(payload[1])
            else:
                current_values = _ensure_numpy(payload)
                current_variances = None

            values = current_values if values is None else values + current_values
            if current_variances is None:
                if variances is not None:
                    variances = None
            else:
                variances = (
                    current_variances
                    if variances is None
                    else variances + current_variances
                )
    elif isinstance(histogram, hist.Hist):
        values = _ensure_numpy(histogram.values(flow=True))
        raw_variances = histogram.variances(flow=True)
        variances = None if raw_variances is None else _ensure_numpy(raw_variances)
    else:
        raise TypeError(f"Unsupported histogram type: {type(histogram)!r}")

    if values is None:
        values = np.array([])

    summary: Dict[str, Any] = {
        "sumw": float(np.sum(values)) if values.size else 0.0,
        "sumw2": float(np.sum(variances)) if variances is not None else None,
        "values": values,
        "variances": variances,
    }
    return summary


TupleKey = Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]


def materialise_tuple_dict(hist_store: Mapping[TupleKey, Any]) -> OrderedDict:
    """Construct an ordered mapping keyed by histogram tuple identifiers."""

    ordered_items = []
    for key, histogram in sorted(hist_store.items(), key=lambda item: item[0]):
        if not isinstance(key, tuple) or len(key) != 5:
            raise ValueError(
                "Histogram accumulator keys must be 5-tuples of (variable, channel, "
                "application, sample, systematic)."
            )
        summary = _summarise_hist(histogram)
        ordered_items.append((key, summary))

    return OrderedDict(ordered_items)


def _tuple_dict_stats(tuple_dict: Mapping[TupleKey, Mapping[str, Any]]) -> Tuple[int, int]:
    total_bins = 0
    filled_bins = 0
    for summary in tuple_dict.values():
        values = _ensure_numpy(summary.get("values", np.array([])))
        total_bins += int(values.size)
        filled_bins += int(np.count_nonzero(values))
    return total_bins, filled_bins

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Execute the example processor and write a cloudpickle file containing "
            "tuple-keyed histogram summaries for downstream training."
        )
    )
    parser.add_argument('jsonFiles'           , nargs='?', default=''           , help = 'Json file(s) containing files and metadata')
    parser.add_argument('--prefix', '-r'     , nargs='?', default=''           , help = 'Prefix or redirector to look for the files')
    parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s'   , default=100000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'   , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--do-errors', action='store_true', help = 'Save the w**2 coefficients')
    parser.add_argument('--split-lep-flavor', action='store_true', help = 'Split up categories by lepton flavor')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')

    args = parser.parse_args()
    jsonFiles        = args.jsonFiles
    prefix           = args.prefix
    nworkers         = int(args.nworkers)
    chunksize        = int(args.chunksize)
    nchunks          = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname          = args.outname
    outpath          = args.outpath
    treename         = args.treename
    do_errors        = args.do_errors
    wc_lst = args.wc_list if args.wc_list is not None else []

    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if sampleName.endswith('.json'): sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]['redirector'] = prefix

    if isinstance(jsonFiles, str) and ',' in jsonFiles:
        jsonFiles = jsonFiles.replace(' ', '').split(',')
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith('/'): jsonFile+='/'
            for f in os.path.listdir(jsonFile):
                if f.endswith('.json'): allInputFiles.append(jsonFile+f)
        else:
            allInputFiles.append(jsonFile)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f'[ERROR] Input file {f} not found!')
        # This input file is a json file, not a cfg
        if f.endswith('.json'):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(' >> Reading json from cfg file...')
                lines = fin.readlines()
                for l in lines:
                    if '#' in l: l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l): prefix = nl
                            else: LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l): prefix = l
                        else: LoadJsonToSampleName(l, prefix)

    flist = {}
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = samplesdict[sname]['year']
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
        samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])
        # Print file info
        print('>> '+sname)
        print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
        print('   - year         : %s'   %samplesdict[sname]['year'])
        print('   - xsec         : %f'   %samplesdict[sname]['xsec'])
        print('   - histAxisName : %s'   %samplesdict[sname]['histAxisName'])
        print('   - options      : %s'   %samplesdict[sname]['options'])
        print('   - tree         : %s'   %samplesdict[sname]['treeName'])
        print('   - nEvents      : %i'   %samplesdict[sname]['nEvents'])
        print('   - nGenEvents   : %i'   %samplesdict[sname]['nGenEvents'])
        print('   - SumWeights   : %f'   %samplesdict[sname]['nSumOfWeights'])
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)


    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]['WCnames']:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        # Yes, why not have the output be in correct English?
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + ' and ' + wc_lst[1]
        else:
            wc_print = ', '.join(wc_lst[:-1]) + ', and ' + wc_lst[-1]
        print('Wilson Coefficients: {}.'.format(wc_print))
    else:
        print('No Wilson coefficients specified')

    processor_instance = simple_processor.AnalysisProcessor(samplesdict,wc_lst,do_errors,)

    exec_instance = processor.FuturesExecutor(workers=nworkers)
    runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)

    # Run the processor and get the output
    tstart = time.time()
    output = runner(flist, treename, processor_instance)
    dt = time.time() - tstart

    histograms_only = {
        key: value
        for key, value in output.items()
        if isinstance(value, (HistEFT, hist.Hist))
    }
    tuple_mapping = materialise_tuple_dict(histograms_only)
    total_bins, filled_bins = _tuple_dict_stats(tuple_mapping)
    fill_fraction = (100 * filled_bins / total_bins) if total_bins else 0.0
    print("Filled %.0f bins, nonzero bins: %1.1f %%" % (total_bins, fill_fraction))
    print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

    # Save the output
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(tuple_mapping, fout)
    print("Done!")
