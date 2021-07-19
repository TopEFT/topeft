import os, sys, argparse, uproot
import subprocess

def isdigit(a):
  ''' Redefinition of str.isdigit() that takes into account negative numbers '''
  if a.isdigit(): return True
  m = a[0]; n = a[1:]
  if m == '-' and n.isdigit(): return True
  return False

def findValidRootfiles(path, sampleName = '', getOnlyNumberedFiles = False, verbose = False, FullPaths = False, retry = True):
  ''' Find rootfiles in path with a given name '''
  files = []
  if ',' in sampleName:
    sl = sampleName.replace(' ', '').split(',')
    return findValidRootfiles(path, sl, getOnlyNumberedFiles, verbose, FullPaths)
  elif isinstance(sampleName, list):
    for s in sampleName:
      files += findValidRootfiles(path, s, getOnlyNumberedFiles, verbose, FullPaths)
      #if len(files) == 0:
      #  files += findValidRootfiles(path, 'Tree_'+s, getOnlyNumberedFiles, verbose, FullPaths)
    return files
  ### Files from a T2 !!
  if path.startswith('root'):

    # Not sure which case this was used for
    #if not sampleName.endswith('.root'): sampleName+='.root'

    # Let's try to get the list of files using xrdfs ls
    if not sampleName.endswith('.root'):
      root_files_full_paths = []
      redirector = path.split("/store")[0] # The redirector should be everything up to the /store
      sample_name = path.split(redirector)[1:][0] # The path should be everything after the redirector
      root_files = (subprocess.run(["xrdfs",redirector,"ls",sample_name],capture_output=True,text=True)).stdout.splitlines() # Get all the files in the dir
      for f in root_files:
        if not f.endswith('.root'): continue # Skip anything that's not a root file
        f = redirector+f
        root_files_full_paths.append(f)
      return root_files_full_paths

    else:
      print('Ok, file in T2...\nReturning: ', [path+sampleName] if FullPaths else [sampleName])
      return [path+sampleName] if FullPaths else [sampleName]

  elif path.upper().startswith('DAS'):
    return GetFileListFromDataset(sampleName, True)
  if not path[-1] == '/': path += '/'
  if verbose: print(' >> Looking for files in path: ' + path)
  for f in os.listdir(path):
    if not f[-5:] == '.root': continue
    #if not '_' in f: continue
    n = f[:-5].split('_')[-1]
    s = f[:-5].split('_')[:-1]
    if not isdigit(n): s.append(n)
    fname = ''
    for e in s: fname+=e+'_'
    if fname[-1] == '_': fname = fname[:-1]
    if getOnlyNumberedFiles and not n.isdigit(): continue
    if sampleName != '' and fname != sampleName and (fname+'_'+n) != sampleName and (fname+'.root') != sampleName and (fname+'_'+n+'.root') != sampleName: continue
    if verbose: print(' >> Adding file: ', f)
    files.append(f)
  if FullPaths: files = [path + x for x in files]
  if len(files) == 0: 
    if retry: files = findValidRootfiles(path, 'Tree_' + sampleName, getOnlyNumberedFiles, verbose, FullPaths, False)
    if len(files) == 0: 
      if os.path.isdir(path+sampleName):
        print('Looking for samples in: ', path+sampleName)
        return GetSampleListInDir(path+sampleName)
      return []
  return files

def GetFiles(path, name, verbose = False):
  ''' Get all rootfiles in path for a given process name'''
  print('Getting files for path and name = [%s, %s]'%(path, name))
  return findValidRootfiles(path, name, False, verbose, FullPaths = True)

def GetListOfWCs(fname):
  ''' Retruns a list of the WC names from WCnames, (retruns [] if not an EFT sample) '''
  wc_names_lst = []
  tree = uproot.open(f'{fname}:Events')
  if 'WCnames' not in tree.keys():
    wc_names_lst = []
  else:
    wc_info = tree['WCnames'].array(entry_stop=1)[0]
    for idx,i in enumerate(wc_info):
      h = hex(i)[2:]                                 # Get rid of the first two characters
      wc_fragment = bytes.fromhex(h).decode('utf-8') # From: https://stackoverflow.com/questions/3283984/decode-hex-string-in-python-3
      # The WC names that are longer than 4 letters are too long to be encoded in a 64-bit integer:
      #   - They're instead stored in two subsequent entries in the list
      #   - This means that the decoded names in wc_info go like this [... 'ctlT' , '-i' ...]
      #   - The leading '-' indicates the given fragment is the trailing end of the previous WC name
      #   - The following logic is supposed to put those fragments back together into the WC name
      if not wc_fragment.startswith("-"):
        wc_names_lst.append(wc_fragment)
      else:
        leftover = wc_fragment[1:]                    # This leftover part of the WC goes with the previous one (but get rid of leading '-')
        wc_names_lst[-1] = wc_names_lst[-1]+leftover  # So append this trailing fragment to the leading framgenet to reconstruct the WC name
  return wc_names_lst

def GetNGenEvents(fname):
  ''' Returns number of events from the 'Count' histograms '''
  if isinstance(fname, list):
    c = 0
    for f in fname: c+=GetNGenEvents(f)
    return c
  elif isinstance(fname, str):
    f = uproot.open(fname)
    h = f['Count']
    return h.values[0]
  else: print('[ERROR] [GetNGenEvents]: wrong input')

def GetSumWeights(fname):
  ''' Returns number of events from the 'SumWeights' histograms '''
  if isinstance(fname, list):
    c = 0
    for f in fname: c+=GetSumWeights(f)
    return c
  elif isinstance(fname, str):
    f = uproot.open(fname)
    h = f['SumWeights']
    return h.values[0]
  else: print ('[ERROR] [GetSumWeights]: wrong input')

def GetEntries(fname, treeName = 'Events'):
  ''' Returns number of events from the tree 'Events' in a file '''
  if isinstance(fname, list):
    c = 0
    for f in fname: c+=GetEntries(f, treeName)
    return c
  elif isinstance(fname, str):
    f = uproot.open(fname)
    t = f[treeName]
    return t.num_entries
  else: print('[ERROR] [GetEntries]: wrong input')

def GuessIsData(fname):
  ''' Guess if a tree is data or simulation '''
  if isinstance(fname, list): fname = fname[0] # Assume all files are the same process/dataset
  f = uproot.open(fname)
  t = f['Events']
  if 'genWeight' in t.keys(): return False
  return True

def guessPathAndName(p):
  ''' Guess path and sample name for a given rootfile '''
  path = ''; n = -1
  while '/' in p:
    path += p[:p.index('/')+1]
    p     = p[p.index('/')+1:]
  if p[-5:] == '.root': p = p[:-5]
  elif os.path.isdir(path + p):
    path = path + p
    p = ''
  if len(path) > 0 and not path[-1] == '/': path += '/'
  if '_' in p: 
    n = p.split('_')[-1]
    s = p.split('_')[:-1]
    if not isdigit(n): 
      s.append(n)
      n = '-1'
    p = ''
    for e in s: p+=e+'_'
    if p[-1] == '_': p = p[:-1]
  return path, p, n

def guessProcessName(fileName):
  ''' Guess the name of the process for a given file '''
  if isinstance(fileName, list): 
    path, name, n = guessPathAndName(fileName[0])
    fileName = name
    if fileName[-5:] == '.root': fileName = fileName[:-5]
    while '/' in fileName: fileName = fileName[fileName.index('/')+1:]
  return fileName

def groupFilesInDic(listOfFiles, name, i=-1, verbose = False):
  ''' Manages a dictionary with sample names and lists of samples '''
  if isinstance(name, list):
    for e in name:
      path, nam, n = guessPathAndName(e)  
      groupFilesInDic(listOfFiles, nam, n)
    return
  fname = name + '_' + str(i) + '.root' if str(i).isdigit() else name + '.root'
  if name in listOfFiles: listOfFiles[name].append(fname)
  else: 
    newList = [fname]
    listOfFiles[name] = newList
    if verbose: print(' >> Sample found: ' + name)

def getDicFiles(inFolder):
  ''' Get a dictionary with sample names and lists of files  '''
  listOfFiles = {}
  files = findValidRootfiles(inFolder)
  groupFilesInDic(listOfFiles,files)
  return listOfFiles
  
def GetAllInfoFromFile(fname, treeName = 'Events'):
  ''' Returns a list with all the info of a file ''' 
  if isinstance(fname, list):
    nEvents = 0
    nGenEvents = 0
    nSumOfWeights = 0
    isData = False
    for f in fname: 
      iE, iG, iS, isData = GetAllInfoFromFile(f, treeName)
      nEvents += iE
      nGenEvents += iG
      nSumOfWeights += iS
    return [nEvents, nGenEvents, nSumOfWeights, isData]
  elif isinstance(fname, str):
    print('Opening with uproot: ', fname)
    f = uproot.open(fname)
    t = f[treeName]
    isData = not 'genWeight' in t#.keys()
    nEvents = int(t.num_entries)
    ## Method 1: from histograms
    if 'Count' in f and False:
      hc = f['Count']
      nGenEvents = hc.values[0] #hc.GetBinContent(1) if isinstance(hc,TH1F) else 1
      nSumOfWeights = 0
      keys = [str(k) for k in f.keys()]
      for k in keys:
        if 'SumWeights' in str(k):
          hs = f['SumWeights']
          nSumOfWeights = hs.values[0]
      if nSumOfWeights == 0: 
        nSumOfWeights = nGenEvents
    # Method 2: from 'Runs' tree
    elif (('Runs' in f) & (not isData)):
      r = f['Runs']
      genEventSumw  = 'genEventSumw'  if 'genEventSumw'  in r else 'genEventSumw_'
      genEventCount = 'genEventCount' if 'genEventCount' in r else 'genEventCount_'
      nGenEvents    = sum(r[genEventCount] .array())
      nSumOfWeights = sum(r[genEventSumw].array())
    # Method 3: from unskimmed file
    else:
      nGenEvents = nEvents
      nSumOfWeights = sum(t['genWeight']) if not isData else nEvents
    return [nEvents, nGenEvents, nSumOfWeights, isData]
  else: print('[ERROR] [GetAllInfoFromFile]: wrong input')

def GetProcessInfo(path, process='', treeName = 'Events'):
  ''' Prints all info from a process in path '''
  if isinstance(path, list): 
    files = path
    path, process, k = guessPathAndName(files[0])
  else: files = GetFiles(path, process)
  nEvents, nGenEvents, nSumOfWeights, isData = GetAllInfoFromFile(files, treeName)
  fileType = '(Data)' if isData else ('(MC)')
  print('\n##################################################################')
  print(' path: ' + path)
  print(' Process:            ' + process + ' ' + fileType)
  print(' Number of files:    ' + str(len(files)))
  print(' Total entries:      ' + str(nEvents))
  if isData:
    print(' Triggered events:   ' + str(nGenEvents))
  else: 
    print(' Generated events:   ' + str(nGenEvents))
    print(' Sum of gen weights: ' + str(nSumOfWeights))
  print('##################################################################\n')

def IsVarInTree(fname, var, treeName = 'Events'):
  ''' Check if a given file and tree contains a branch '''
  if not os.path.isfile(fname):
    print('ERROR: %s does not exists!'%fname)
    return False
  f = uproot.open(fname)
  t = f[treeName]
  return var in t.keys()

def GetValOfVarInTree(fname, var, treeName = 'Events'):
  ''' Check the value of a var in a tree '''
  if not os.path.isfile(fname):
    print('ERROR: %s does not exists!'%fname)
    return False
  f = uproot.open(fname)
  t = f[treeName]
  return t[var][0]


##################################
# Extra functions to work check .root files from terminal

def main():
 # Parsing options
 path = './'
 sample = ''
 pr = argparse.ArgumentParser()
 pr.add_argument('path', help='Input folder', type = str)
 pr.add_argument('--sample', type = str, default = '')
 pr.add_argument('-i','--inspect', action='store_true', help='Print branches')
 pr.add_argument('-t','--treeName', default='Events', help='Name of the tree')
 pr.add_argument('-c','--cfg', default='tempsamples', help='Name of the output cfg file')
 pr.add_argument('-o','--options', default='', help='Options')
 pr.add_argument('-p','--prod','--prodName', default='', help='Name of the production')
 pr.add_argument('-v','--verbose', action='store_true', help='Verbose')
 pr.add_argument('-x','--xsec', '--xsecfile', default='cfg/xsec.cfg', help='xsec file')
 args = pr.parse_args()
 if args.sample:  sample = args.sample
 treeName = args.treeName
 printb = args.inspect
 path = args.path
 prod = args.prod
 if os.path.isdir(path) and not path[-1] == '/': path += '/'

 if prod != '':
   CreateCfgFromCrabOutput(path, prod, args.cfg, args.xsec, args.verbose)
   exit()
 if sample == '':
   origpath = path
   path, sample, n = guessPathAndName(path)

   if sample == '': 
     # Only one path is given... reading files in path and doing nothing
     d = getDicFiles(path)
     for c in d:
       print(' >> ' + c + ': ', d[c])

   else:
     totfile = path + sample + '_' + n + '.root' if int(n) >= 0 else path + sample + '.root'
     if os.path.isfile(totfile): 
       GetProcessInfo([totfile], treeName = treeName)
       exit()
     else:
       GetProcessInfo(path, sample, treeName)
 else:
   GetProcessInfo(path, sample, treeName)
   exit()

##############################################
### From crab output

def GetSampleListInDir(dirname):
  filelist = []
  for path, subdirs, files in os.walk(dirname):
    for name in files:
      if not name.endswith('.root'): continue
      fname = '%s/%s'%(path,name)
      filelist.append(fname)
  return filelist

def CraftSampleName(name):
  # Deal with 'ext' in the end
  if   'ext' in name[-3:]: name = name[:-3] + '_' + name[-3:]
  elif 'ext' in name[-4:]: name = name[:-4] + '_' + name[-4:]
  # Rename bits...
  name = name.replace('madgraphMLM', 'MLM')
  name = name.replace('ST_tW_top', 'tW')
  name = name.replace('ST_tW_antitop', 'tbarW')
  name = name.replace('NoFullyHadronicDecays', 'noFullHad')
  # Delete bits...
  deleteWords = ['13TeV', 'powheg', 'Powheg', 'pythia8']
  s = name.replace('-', '_').split('_')
  a = ''
  for bit in s:
    if bit in deleteWords: continue    
    else: a += '_' + bit
  if a.startswith('_'): a = a[1:]
  if a.endswith('_')  : a = a[:-1]
  while '__' in a: a = a.replace('__', '_')
  return a

def haddProduction(dirname, prodname, verbose=0):
  dirnames = []
  samplenames = []
  n = len(prodname)
  for path, subdirs, files in os.walk(dirname):
    for s in subdirs:
      if not s.startswith(prodname+'_'): continue
      else:
        treeName = s[n+1:]
        dirName  = path + '/' + s
        sname = dirName[len(dirname):]
        sname.replace('//', '/')
        sampleName = CraftSampleName(treeName)
        dirnames.append(sname)#dirName)
        samplenames.append(sampleName)
        if verbose >= 1: print(' >> Found sample: ' + pcol.red + treeName + pcol.white + ' (' + pcol.cyan + sampleName + pcol.white + ')' + pcol.end)
  return [dirnames, samplenames]

def CreateCfgFromCrabOutput(dirname, prodname, out='samples', xsecfile='cfg/xsec.cfg', verbose=0):
  if not out.endswith('.cfg'): out += '.cfg'
  f = open(out, 'a+')
  nlines = len(f.readlines())
  if nlines <= 1:
    f.write('path: %s\n'%dirname)
    f.write('xsec: %s\n'%xsecfile)
    f.write('verbose: %i\n'%int(verbose))
  dirs, samples = haddProduction(dirname, prodname)
  f.write('\n\n')
  for d, s in zip(dirs, samples):
    f.write('%s : %s\n'%(s, d))
  print('Created file: %s'%out)

def GetWCnames(files, treeName='Events'):
  ''' To be done: function to fetch WC names from nanoAOD sample '''
  if isinstance(files, list): fname = files[0]
  return []
  


if __name__ == '__main__':
  main()
