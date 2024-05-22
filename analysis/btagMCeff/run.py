#!/usr/bin/env python
import json
import time
import cloudpickle
import gzip
import os

import numpy as np
from coffea import hist, processor
from coffea.util import load
from coffea.nanoevents import NanoAODSchema

import btagMCeff

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s'   , default=500000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='btagMCeff', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'   , default='Events', help = 'Name of the tree inside the files')

    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    dotest     = args.test
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname    = args.outname
    outpath    = args.outpath
    treename   = args.treename

    if dotest:
        nchunks = 2
        chunksize = 10000
        nworkers = 1
        print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))

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
                    if '#' in l:
                        l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)


    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = samplesdict[sname]['year']
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        nevts_total += samplesdict[sname]['nEvents']
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
        print('   - SumWeights   : %i'   %samplesdict[sname]['nSumOfWeights'])
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)

    processor_instance = btagMCeff.AnalysisProcessor(samplesdict)

    executor = processor.futures_executor(workers=nworkers)
    runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)

    tstart = time.time()
    output = runner(flist, treename, processor_instance)
    dt = time.time() - tstart

    nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))
    print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

    # This is taken from the DM photon analysis...
    # Pickle is not very fast or memory efficient, will be replaced by something better soon
    #    with lz4f.open("pods/"+options.year+"/"+dataset+".pkl.gz", mode="xb", compression_level=5) as fout:
    if not outpath.endswith('/'): outpath += '/'
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    print('Saving output in %s...'%(outpath + outname + ".pkl.gz"))
    with gzip.open(outpath + outname + ".pkl.gz", "wb") as fout:
        cloudpickle.dump(output, fout)
    print('Done!')




