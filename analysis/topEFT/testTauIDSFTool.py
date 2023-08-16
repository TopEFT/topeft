#! /usr/bin/env python
# Author: Izaak Neutelings (July 2019)
# Usage:
#   ./test/testTauIDSFTool.py
from __future__ import print_function
import time; start0 = time.time()
from TauPOG.TauIDSFs.TauIDSFTool import TauIDSFTool, TauESTool, TauFESTool
import json
start1 = time.time()

def green(string,**kwargs): return "\x1b[0;32;40m%s\033[0m"%string

def printSFTable(year,id,wp,vs='pt',emb=False,otherVSlepWP=False):
  assert vs in ['pt','dm','ptdm','eta'], "'vs' argument should be 'pt', 'dm', 'ptdm', or 'eta'!"
  dm = (vs=='dm')
  if emb and 'VSjet' not in id:
      print("SFs for ID '%s' not available for embedded samples. Skipping..."%id)
      return
  sftool = TauIDSFTool(year,id,wp,dm=dm,emb=emb,otherVSlepWP=otherVSlepWP)
  testool = TauESTool(year,id)
  oldYear = ''
  if year == 'UL2016_preVFP' or year == 'UL2016_postVFP':
    oldYear = '2016Legacy'                                                                                                                                                                                
  elif year == 'UL2017':
    oldYear = '2017ReReco'
  elif year == 'UL2018':
    oldYear = '2018ReReco' 
  antiEleSFTool = TauIDSFTool(oldYear,'antiEleMVA6','Loose')
  antiMuSFTool  = TauIDSFTool(oldYear,'antiMu3','Tight')
  #festool = TauFESTool(year, id)
  if vs=='ptdm':
      SFdic = {}
      #SFdic["SF"]  = {}
      #SFdic["TES"] = {}
      #SFdic["FES"] = {}
      #SFdic["Fake"] = {}
      #ptvals = [10,20,50,100,140,200]
      ptvals = []
      i = 10
      while i <= 200:
        ptvals.append(i)
        i += 5
      dmvals = [0,1,10,11]
      year_=year
      if year_.startswith('UL'): year_=year_[2:]
      uncerts=['uncert0','uncert1','syst_alleras','syst_%s' % year_, 'syst_dmX_%s' % year_]
      etavals = []
      i = 0.0
      while i <= 2.4:
        etavals.append(i)
        i += 0.2
      #etavals = [0,0.2,0.5,1.0,1.5,2.0,2.2,2.3,2.4]
      for eta in etavals:
        etaKey = "eta:[" + str(eta) + "," + str(eta+0.2) + "]"
        #SFdic[etaKey] = {}
        genvals = [1.0, 2.0]
        for gen in genvals:
          genKey = "gen:[" + str(gen) + "," + str(gen) + "]"
          #SFdic[etaKey][genKey] = {}
          #if gen == 1.0:
          #  SFdic[etaKey][genKey]["value"] = antiEleSFTool.getSFvsEta(eta,1)
          #elif gen == 2.0:
          #  SFdic[etaKey][genKey]["value"] = antiMuSFTool.getSFvsEta(eta,2)
      #    print(genmatch)
      #    SFdic["Fake"][eta].append(sftool.getSFvsEta(eta,genmatch))
      #    SFdic["Fake"][eta].append(sftool.getSFvsEta(eta,genmatch,"Up"))
      #    SFdic["Fake"][eta].append(sftool.getSFvsEta(eta,genmatch,"Down"))
        #for dm in dmvals:
        #  if dm == 11:
        #    continue
        #  SFdic["FES"][eta][dm] = []
        #  SFdic["FES"][eta][dm].append(festool.getFES(eta,dm,1))
        #  SFdic["FES"][eta][dm].append(festool.getFES(eta,dm,1,"Up"))
        #  SFdic["FES"][eta][dm].append(festool.getFES(eta,dm,1,"Down"))
      for dm in dmvals:
        dm = float(dm)
        dmKey = "dm:[" + str(dm) + "," + str(dm) + "]"
        SFdic[dmKey] = {}
        #SFdic["TES"][pt] = {}
        for pt in ptvals:
          ptKey = "pt:[" + str(pt) + "," + str(pt+5) + "]"
          SFdic[dmKey][ptKey] = {}
          #SFdic[dmKey][ptKey]["value"] = sftool.getSFvsDMandPT(pt,dm,5)
          #SFdic[dmKey][ptKey]["up"]   = sftool.getSFvsDMandPT(pt,dm,5,'uncert0_up')
          #SFdic[dmKey][ptKey]["down"] = sftool.getSFvsDMandPT(pt,dm,5,'uncert0_down')
          #SFdic["TES"][pt][dm] = []
          SFdic[dmKey][ptKey]["value"] = testool.getTES(pt,dm,5)
          SFdic[dmKey][ptKey]["up"]    = testool.getTES(pt,dm,5,'uncert0_up')
          SFdic[dmKey][ptKey]["down"]  = testool.getTES(pt,dm,5,'uncert0_down')
          
        print(">>> ")
        print(">>> SF for %s WP of %s in %s with pT = %s GeV"%(wp,green(id),year,pt))
        print(">>> ")
        print(">>> %20s"%('var \ DM')+''.join("%9d"%dm for dm in dmvals))
        print(">>> %20s"%("central") +''.join("%9.5f"%sftool.getSFvsDMandPT(pt,dm,5)    for dm in dmvals))
        for u in uncerts: 
          
          print(">>> %20s"%(u+"_up")      +''.join("%9.5f"%sftool.getSFvsDMandPT(pt,dm,5,u.replace('dmX','dm%s' % dm)+'_up')   for dm in dmvals))
          print(">>> %20s"%(u+"_down")    +''.join("%9.5f"%sftool.getSFvsDMandPT(pt,dm,5,u.replace('dmX','dm%s' % dm)+'_down') for dm in dmvals))
        print(">>> ")
      return SFdic
  elif vs=='pt':
      ptvals = [10,20,21,25,26,30,31,35,40,50,70,100,200,500,600,700,800,1000,1500,2000,]
      print(">>> ")
      print(">>> SF for %s WP of %s in %s"%(wp,green(id),year))
      print(">>> ")
      print(">>> %10s"%('var \ pt')+''.join("%9.1f"%pt for pt in ptvals))
      print(">>> %10s"%("central") +''.join("%9.5f"%sftool.getSFvsPT(pt,5)        for pt in ptvals))
      print(">>> %10s"%("up")      +''.join("%9.5f"%sftool.getSFvsPT(pt,5,'Up')   for pt in ptvals))
      print(">>> %10s"%("down")    +''.join("%9.5f"%sftool.getSFvsPT(pt,5,'Down') for pt in ptvals))
      print(">>> ")
      ###sftool.getSFvsDM(25,1,5)   # results in an error
      ###sftool.getSFvsEta(1.5,1,5) # results in an error
  elif vs=='dm':
    dmvals = [0,1,5,6,10,11]
    for pt in [25,50]:
      print(">>> ")
      print(">>> SF for %s WP of %s in %s with pT = %s GeV"%(wp,green(id),year,pt))
      print(">>> ")
      print(">>> %10s"%('var \ DM')+''.join("%9d"%dm for dm in dmvals))
      print(">>> %10s"%("central") +''.join("%9.5f"%sftool.getSFvsDM(pt,dm,5)        for dm in dmvals))
      print(">>> %10s"%("up")      +''.join("%9.5f"%sftool.getSFvsDM(pt,dm,5,'Up')   for dm in dmvals))
      print(">>> %10s"%("down")    +''.join("%9.5f"%sftool.getSFvsDM(pt,dm,5,'Down') for dm in dmvals))
      print(">>> ")
      ###sftool.getSFvsPT(pt,5)     # results in an error
      ###sftool.getSFvsEta(1.5,1,5) # results in an error
  elif vs=='eta':
    if emb:
      print("vsEta binned SFs not available for embedded samples. Skipping...")
      return
    etavals = [0,0.2,0.5,1.0,1.5,2.0,2.2,2.3,2.4]
    for genmatch in [1,2]:
      print(">>> ")
      print(">>> SF for %s WP of %s in %s with genmatch %d"%(wp,green(id),year,genmatch))
      print(">>> ")
      print(">>> %10s"%('var \ eta')+''.join("%9.3f"%eta for eta in etavals))
      print(">>> %10s"%("central")  +''.join("%9.5f"%sftool.getSFvsEta(eta,genmatch)        for eta in etavals))
      print(">>> %10s"%("up")       +''.join("%9.5f"%sftool.getSFvsEta(eta,genmatch,'Up')   for eta in etavals))
      print(">>> %10s"%("down")     +''.join("%9.5f"%sftool.getSFvsEta(eta,genmatch,'Down') for eta in etavals))
      print(">>> ")
      ###sftool.getSFvsPT(pt,5)     # results in an error
      ###sftool.getSFvsEta(1.5,1,5) # results in an error
  

def printTESTable(year,id):
  testool = TauESTool(year,id)
  ptvals  = [25,102,175] #[25,30,102,170,175]
  dmvals  = [0,1,5,10,11]
  for pt in ptvals:
    print(">>> ")
    print(">>> TES for '%s' ('%s') and pT = %s GeV"%(green(id),year,pt))
    print(">>> ")
    print(">>> %10s"%('var \ DM')+''.join("%9d"%dm for dm in dmvals))
    print(">>> %10s"%("central") +''.join("%9.5f"%testool.getTES(pt,dm,5)        for dm in dmvals))
    print(">>> %10s"%("up")      +''.join("%9.5f"%testool.getTES(pt,dm,5,'Up')   for dm in dmvals))
    print(">>> %10s"%("down")    +''.join("%9.5f"%testool.getTES(pt,dm,5,'Down') for dm in dmvals))
    print(">>> ")
  

def printFESTable(year):
  testool = TauFESTool(year)
  etas    = [0.5,2.0]
  dmvals  = [0,1,10]
  for eta in etas:
    print(">>> ")
    print(">>> TES for eta = %.1f in '%s'"%(eta,year))
    print(">>> ")
    print(">>> %10s"%('var \ DM')+''.join("%9d"%dm for dm in dmvals))
    print(">>> %10s"%("central") +''.join("%9.5f"%testool.getFES(eta,dm,1)        for dm in dmvals))
    print(">>> %10s"%("up")      +''.join("%9.5f"%testool.getFES(eta,dm,1,'Up')   for dm in dmvals))
    print(">>> %10s"%("down")    +''.join("%9.5f"%testool.getFES(eta,dm,1,'Down') for dm in dmvals))
    print(">>> ")
  

if __name__ == "__main__":  
  print(">>> ")
  print(">>> start test tau ID SF tool")
  
  testIDTool   = True #and False
  testTESTool  = True and False
  testFESTool  = True and False
  emb          = True and False
  otherVSlepWP = True and False
  
  start2       = time.time()
  years        = [
    #'2016Legacy',
    #'2017ReReco',
    #'2018ReReco',
    'UL2016_preVFP',
    'UL2016_postVFP',
    'UL2017',
    'UL2018',
  ]
  tauIDs      = [
    #'MVAoldDM2017v2',
    'DeepTau2017v2p1VSjet',
    #'antiEleMVA6',
    #'antiMu3',
    #'DeepTau2017v2p1VSmu',
    #'DeepTau2017v2p1VSe'
  ]
  tauESs = [
    #'MVAoldDM2017v2',
    'DeepTau2017v2p1VSjet',
  ]
  WPs         = [
    #'VLoose',
    #'Loose',
    #'Medium',
    'Tight',
    #'VTight'
  ]
  SFdic = {}
  for year in years:
    SFdic[year] = {}
    for id in tauIDs:
      vslist = ['eta'] if any(s in id for s in ['anti','VSe','VSmu']) else (['pt','dm'] if emb else ['ptdm'])
      for vs in vslist:
        for wp in WPs:
          if 'antiMu' in id and wp=='Medium': continue
          if testIDTool:
            if wp == "Loose":
              wpFlag = "wp:[4,7]"
            if wp == "Medium":
              wpFlag = "wp:[8:15]"
            if wp == "Tight":
              wpFlag = "wp:[16:999]"
            #SFdic[year][wp] = {}
            #SFdic[year][wp]["Tau_SF"] = {}
            #SFdic[year][wp]["Tau_SF"]["dm_pt"] = printSFTable(year,id,wp,vs,emb,otherVSlepWP)
            #SFdic[year]["Tau_SF"] = {}
            #SFdic[year]["Tau_SF"]["eta_gen"] = printSFTable(year,id,wp,vs,emb,otherVSlepWP)
            SFdic[year]["Tau_TES"] = {}
            SFdic[year]["Tau_TES"]["dm_pt"] = printSFTable(year,id,wp,vs,emb,otherVSlepWP)
            print("Succeeded with ", year, id)
            print("Current keys ", SFdic.keys())
            fileName = "TauTES"+year+".json"
            json_object = json.dumps(SFdic[year], indent=4)
            with open(fileName, "w") as outfile:
              outfile.write(json_object)
    if testTESTool:
      for id in tauESs:
        printTESTable(year,id)
    if testFESTool:
      printFESTable(year)
  
  #json_object = json.dumps(SFdic, indent=4)
  #with open("sample.json", "w") as outfile:
  #  outfile.write(json_object)
  start3 = time.time()
  print(">>> ")
  print(">>> done after %.1f seconds (%.1f for imports, %.1f for loops)"%(time.time()-start0,start1-start0,start3-start2))
  
