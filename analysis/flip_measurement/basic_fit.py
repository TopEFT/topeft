import pickle
import gzip
import numpy as np
import uproot
import ROOT
ROOT.gROOT.SetBatch(True)

#Load hists from pickle file created by TopEFT
hists={}
fin = 'histos/flipTopEFT.pkl.gz'

with gzip.open(fin) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
        if k in hists: hists[k]+=hin[k]
        else:               hists[k]=hin[k]


# Bit of a hacky way, since ROOT and RooFit is easier to use than SciPy for fitting Gaussian + exponential
# Convert hist to TH1
fout = uproot.recreate("hists.root")
fout['osz'] = hists['invmass'][{'process': sum, 'channel': 'osz'}].to_numpy()
osz = fout['osz'].to_pyroot()

# Workspace for fitting
w = ROOT.RooWorkspace("w")
# Crate model for fitting (Gaus1 + Gaus2 + expo) (using RooFit factory syntax)
w.factory(f"SUM::model(nsig[0, {osz.Integral()}]*Gaussian::sig(mass[50, 150], mean[91, 60, 120], sigma[1, 0.1, 20]), nsig1[0, {osz.Integral()}]*Gaussian::sig1(mass, mean, sigma1[1, 0.1, 20]), nbkg[0, {osz.Integral()}]*Exponential::bkg(mass, lambda[-10, 10]))")
# Import data into ws
data = ROOT.RooDataHist("data", "data", w.var("mass"), ROOT.RooFit.Import(osz))
# Fit the model to the data
w.pdf("model").fitTo(data, ROOT.RooFit.PrintLevel(0)) # PrintLevel(0) suppresses printouts. Change to see things like fit results and correlation martrix

# Frame for drawing
f = w.var("mass").frame()

# Plot the data
c1 = ROOT.TCanvas()
c1.cd()
data.plotOn(f)
w.pdf("model").plotOn(f)
w.pdf("model").plotOn(f, ROOT.RooFit.Components(w.pdf("bkg")), ROOT.RooFit.LineStyle(ROOT.kDashed))
f.Draw()
c1.SaveAs("fit_osz.png")

# Compute total signal (sum of both Gaussian terms)
nsig = w.var("nsig").getVal() + w.var("nsig1").getVal()
error = np.sqrt(np.sum(np.square([w.var("nsig").getError(), w.var("nsig1").getError()])))
print(f'Total opposite-signed signal = {nsig:.0f} +/- {error:.2f}')
