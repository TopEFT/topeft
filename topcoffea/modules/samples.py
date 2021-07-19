'''
  samples.py

  This script is executed with a .cfg file as input. The script takes a path and sample names as input (and some other options)
  and creates a dictionary with all the per-sample information used in the analysis.

  - The cross section is read from another .cfg file, based on the name of the process
  - The sum of weights (used for normalization) is calculated from a histogram within the rootfiles
  - The dictorionary contains all samples for the form sample1name_[number].root, grouped into processes
  - Reads the number of entries and, according to the present branches, determines if a sample contains data or MC events

  Returns a dictionary containting all the info for each sample

  Usage:
    >> python samples.py configFile.cfg

  Example of how to run this script in this repo: 
    >> python moca/samples.py cfg/2018.cfg

'''

import os, sys
from coffea.util import save
from topcoffea.modules.DASsearch import GetDatasetFromDAS
from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.fileReader import GetFiles, GetAllInfoFromFile
basepath = topcoffea_path("") # Just want path to topcoffea/topcoffea, not any particular file within it, so just pass "" to the function

def FindFileInDir(fname, dname = '.'):
  if not os.path.isfile(dname+'/'+fname):
    l = list(filter(lambda x: x[0] == fname, [x.split('.') for x in os.listdir(dname)]))
    if len(l) == 0: return False
    else          : l = l[0]
    fname = l[0] + '.' + l[1]
    return fname
  else: return dname+'/'+fname

def loadxsecdic(fname, verbose):
  xsecdir = {}
  dname = '.'
  filename = FindFileInDir(fname, dname)
  if not filename: filename = FindFileInDir(fname, basepath)
  if not filename:
      print('ERROR: not found file %s with cross sections...'%fname)
      return
  if verbose: print(' >> Reading cross section from %s...'%filename)
  f = open(filename)
  lines = f.readlines()
  for l in lines:
    l = l.replace(' ', '')
    l = l.replace('\n', '')
    if l.startswith('#'): continue
    if not ':' in l: continue
    if '#' in l: l = l.split('#')[0]
    if l == '': continue
    lst = l.split(':')
    key = lst[0]
    val = lst[1]
    if val == '': val = 1
    xsecdir[key] = float(val)
  return xsecdir

def GetXsec(xsec, s, verbose, isdata):
  if isdata: return 1
  if isinstance(xsec, int): xsec = float(xsec)
  if isinstance(xsec, str):
    xsecdic = loadxsecdic(xsec, verbose)
    if not s in xsecdic.keys():
      print('ERROR: not found xsec value for sample %s'%s)
      xsec = 1
    else: xsec = xsecdic[s]
  return xsec

def GetSampleList(path, sample):
  dic = getDicFiles(path)
  nfileInPath = len(dic)
  if verbose: print('Found %i files in path %s'%(nfileInPath, path))
  samples = []
  for s in sample:
    dk = dic.keys()
    if not s in dk: s = prefix+'_'+s
    if not s in dk:
      print('WARNING: file %s not in path %s'%(s, path))
    else:
      samples += dic[s]
  return samples

def GetOptions(path, sample, options = ""):
  if not path.endswith('/'): path += '/'
  if not sample.endswith(".root"): sample += '.root'
  #doPUweight  = 'PUweight,' if IsVarInTree(path+sample, 'puWeight') else ''
  #doJECunc    = 'JECunc,'   if IsVarInTree(path+sample, 'Jet_pt_jesTotalUp') else ''
  #doIFSR      = 'doIFSR,'   if IsVarInTree(path+sample, 'nPSWeight') and GetValOfVarInTree(path+sample, 'nPSWeight') == 4 else ''
  #useJetPtNom = 'JetPtNom,' if IsVarInTree(path+sample, 'Jet_pt_nom') else ''
  #options += doPUweight + doJECunc + doIFSR + useJetPtNom + options
  if options.endswith(','): options = options[:-1]
  return options

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Create dict with files and options')
  parser.add_argument('cfgfile'           , default=''           , help = 'Config file with dataset names')
  parser.add_argument('--pretend','-p'    , action='store_true'  , help = 'Create the files but not send the jobs')
  parser.add_argument('--test','-t'       , action='store_true'  , help = 'Sends only one or two jobs, as a test')
  parser.add_argument('--verbose','-v'    , action='store_true'  , help = 'Activate the verbosing')
  parser.add_argument('--path'            , default=''           , help = 'Path to look for nanoAOD')
  parser.add_argument('--sample','-s'     , default=''           , help = 'Sample(s) to process')
  parser.add_argument('--xsec','-x'       , default='xsec'       , help = 'Cross section')
  parser.add_argument('--year','-y'       , default=-1           , help = 'Year')
  parser.add_argument('--options'         , default=''           , help = 'Options to pass to your analysis')
  parser.add_argument('--treename'        , default='Events'     , help = 'Name of the tree')
  parser.add_argument('--nFiles'          , default=None         , help = 'Number of max files (for the moment, only applies for DAS)')

  args, unknown = parser.parse_known_args()
  cfgfile     = args.cfgfile
  verbose     = args.verbose
  pretend     = args.pretend
  dotest      = args.test
  sample      = args.sample
  path        = args.path
  options     = args.options
  xsec        = args.xsec
  year        = args.year
  treeName    = args.treename

  samplefiles = {}
  fileopt = {}
  xsecdic = {}
  sampdic = {}

  if not os.path.isfile(cfgfile) and os.path.isfile(cfgfile+'.cfg'): cfgfile+='.cfg'
  f = open(cfgfile)
  lines = f.readlines()
  for l in lines:
    l = l.replace(' ', '')
    l = l.replace('\n', '')
    if l.startswith('#'): continue
    if '#' in l: l = l.split('#')[0]
    if l == '': continue
    if l.endswith(':'): l = l[:-1]
    if not ':' in l:
      if l in ['path', 'verbose', 'pretend', 'test', 'options', 'xsec', 'year', 'treeName']: continue
      else: samplefiles[l]=l
    else:
      lst = l.split(':')
      key = lst[0]
      val = lst[1] if lst[1] != '' else lst[0]
      if   key == 'pretend'   : pretend   = 1
      elif key == 'verbose'   : verbose   = int(val) if val.isdigit() else 1
      elif key == 'test'      : dotest    = 1
      elif key == 'path'      :
        path      = val
        if len(lst) > 2: 
          for v in lst[2:]: path += ':'+v
      elif key == 'options'   : options   = val
      elif key == 'xsec'      : xsec      = val
      elif key == 'year'      : year      = int(val)
      elif key == 'treeName'  : treeName  = val
      else:
        fileopt[key] = ''#options
        if len(lst) >= 3: fileopt[key] += lst[2]
        samplefiles[key] = val

  # Re-assign arguments...
  aarg = sys.argv
  if '--pretend' in aarg or '-p' in aarg : pretend     = args.pretend
  if '--test'    in aarg or '-t' in aarg : dotest      = args.test
  if args.path       != ''       : path        = args.path
  if args.options    != ''       : options     = args.options
  if args.xsec       != 'xsec'   : xsec        = args.xsec
  if args.year       != -1       : year        = args.year
  if args.treename   != 'Events' : treeName    = args.treename
  if args.verbose    != 0        : verbose     = int(args.verbose)
  xsecdic = loadxsecdic(xsec, verbose)

  for sname in samplefiles.keys():
    sampdic[sname] = {}
    sampdic[sname]['xsec']       = xsecdic[sname] if sname in xsecdic.keys() else 1
    sampdic[sname]['year']       = year
    sampdic[sname]['treeName']   = treeName
    if 'DAS' in options:
      dataset = samplefiles[sname]
      nFiles = int(fileopt[sname]) if fileopt[sname]!='' else None
      #dicFiles = GetDatasetFromDAS(dataset, nFiles, options='file', withRedirector='root://cms-xrd-global.cern.ch/')
      dicFiles = GetDatasetFromDAS(dataset, nFiles, options='file', withRedirector=path)
      nEvents, nGenEvents, nSumOfWeights, isData = GetAllInfoFromFile(dicFiles['files'], sampdic[sname]['treeName'])
      files          = dicFiles['files']
      nEvents        = dicFiles['events']
      fileOptions = ''
    else:
      files = GetFiles(path, samplefiles[sname])
      nEvents, nGenEvents, nSumOfWeights, isData = GetAllInfoFromFile(files, sampdic[sname]['treeName'])
      extraOption = GetOptions(path, files[0].split('/')[-1])
      fileOptions = fileopt[sname]+','+extraOption
    sampdic[sname]['options']    = fileOptions
    sampdic[sname]['files']      = files
    sampdic[sname]['nEvents']       = nEvents
    sampdic[sname]['nGenEvents']    = nGenEvents
    sampdic[sname]['nSumOfWeights'] = nSumOfWeights
    sampdic[sname]['isData']        = isData

  if verbose:
    for sname in samplefiles.keys():
      print('>> '+sname)
      print('   - isData?    : %s'   %('YES' if sampdic[sname]['isData'] else 'NO'))
      print('   - year       : %i'   %sampdic[sname]['year'])
      print('   - xsec       : %1.3f'%sampdic[sname]['xsec'])
      print('   - options    : %s'   %sampdic[sname]['options'])
      print('   - tree       : %s'   %sampdic[sname]['treeName'])
      print('   - nEvents    : %i'   %sampdic[sname]['nEvents'])
      print('   - nGenEvents : %i'   %sampdic[sname]['nGenEvents'])
      print('   - SumWeights : %i'   %sampdic[sname]['nSumOfWeights'])
      print('   - nFiles     : %i'   %len(sampdic[sname]['files']))
      for fname in sampdic[sname]['files']: print('     %s'%fname)
  save(sampdic, '.samples.coffea')

  return sampdic

if __name__ == '__main__':
  main()

