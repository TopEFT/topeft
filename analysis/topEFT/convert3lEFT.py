import pickle
from coffea import hist
import topcoffea.modules.HistEFT as HistEFT
import gzip
import matplotlib.pyplot as plt
import ROOT
ROOT.gSystem.Load('Utils/WCFit_h.so')
ROOT.gSystem.Load('Utils/TH1EFT_cc.so')

'''
Example of converting HistEFT 3l MET plot to TH1EFT
'''

def Serialize():
    hists = {}
    fin = 'histos/plotsTopEFT.pkl.gz'
    print('loading {}'.format(fin))
    with gzip.open(fin) as fin:
      hin = pickle.load(fin)
      for k in hin.keys():
        if k in hists: hists[k]+=hin[k]
        else:               hists[k]=hin[k]
    
    h = hists['met']
    ch3l = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
    h = h.integrate('channel', ch3l)
    h = h.integrate('cut', 'base')
    h = h.sum('sample')
    print('building WCFit')
    h.SetWCFit()
    heft = ROOT.TH1EFT('h', 'h', 40, 0, 200)
    for wc in list(h.WCFit.values())[0]:
        print(wc.GetTag())
        wc_dict = wc.Serialize()
        tag = wc_dict['tag']
        names = wc_dict['names']
        coeffs = wc_dict['coeffs']
        err_coeffs = wc_dict['errs']
        vname = ROOT.vector('string')(0)
        wc = ROOT.WCFit.WCFit()
        wc.setTag(tag)
        for n in names:
            vname.push_back(str(n)) #store WC names in vector
            wc.extend(str(n)) #extend WCFit (fills std::vectors in WCFit class)
        vname = ROOT.vector('string')(0)
        vvalue = ROOT.vector('float')(0)
        for k,v in coeffs:
            vname.push_back('*'.join([names[k[0]],names[k[1]]])) #create pairs of WC names (e.g. 'sm*ctW')
            vvalue.push_back(float(v)) #store WC values in vector
        evalue = ROOT.vector('float')(0)
        for _,v in err_coeffs:
            evalue.push_back(float(v)) #store WC errors in vector
        wc.deserialize(vname, vvalue, evalue) #deserialize name pairs, WC values, error values
        heft.SetBinFit(int(tag), wc)
        heft.SetBinContent(int(tag), wc.evalPoint(ROOT.WCPoint('smpt')))
        heft.SetBinError(int(tag), wc.evalPointError(ROOT.WCPoint('smpt')))
    fout = ROOT.TFile("wc.root", "recreate")
    heft.SetDirectory(fout)
    fout.Write()
    fout.Close()

if __name__ == "__main__":
    Serialize()
