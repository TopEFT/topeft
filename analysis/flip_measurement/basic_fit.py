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
fout = uproot.recreate("hists.root")
fout['ssz'] = hists['invmass'][{'process': sum, 'channel': 'ssz'}].to_numpy()
ssz = fout['ssz'].to_pyroot()

# Workspace for fitting
w = ROOT.RooWorkspace("w")
# Crate model for fitting (Crystal Ball + Gaus + expo) (using RooFit factory syntax)
# Crystal Ball does a bit better at capturing the asymmetric shape due to radiation effects
w.factory(f"SUM::model(nsig[0, {osz.Integral()}]*CBShape::sig(mass[50, 150], mean[91, 60, 120], sigma[1, 0.1, 20], alpha[1, 0, 5], n[5, 0, 10]), nsig1[0, {osz.Integral()}]*Gaussian::sig1(mass, mean, sigma1[1, 0.1, 20]), nbkg[0, {osz.Integral()}]*Exponential::bkg(mass, lambda[-10, 10]))")
#TODO find a better model for same-signed (ss)
w.factory(f"SUM::model_ss(nsig_ss[0, {ssz.Integral()}]*CBShape::sig_ss(mass, mean_ss[91, 60, 120], sigma_ss[1, 0.1, 20], alpha_ss[1, 0, 5], n_ss[5, 0, 10]), nsig1_ss[0, {ssz.Integral()}]*Gaussian::sig1_ss(mass, mean, sigma1_ss[1, 0.1, 20]), nbkg_ss[0, {ssz.Integral()}]*Exponential::bkg_ss(mass, lambda_ss[-10, 10]))")
# Crate model for fitting (Gaus1 + Gaus2 + expo) (using RooFit factory syntax)
#w.factory(f"SUM::model(nsig[0, {osz.Integral()}]*Gaussian::sig(mass[50, 150], mean[91, 60, 120], sigma[1, 0.1, 20]), nsig1[0, {osz.Integral()}]*Gaussian::sig1(mass, mean, sigma1[1, 0.1, 20]), nbkg[0, {osz.Integral()}]*Exponential::bkg(mass, lambda[-10, 10]))")
# Import data into ws
data = ROOT.RooDataHist("data", "data", w.var("mass"), ROOT.RooFit.Import(osz))
data_ss = ROOT.RooDataHist("data_ss", "data_ss", w.var("mass"), ROOT.RooFit.Import(ssz))
# Fit the model to the data
w.pdf("model").fitTo(data, ROOT.RooFit.PrintLevel(0)) # PrintLevel(0) suppresses printouts. Change to see things like fit results and correlation martrix
w.Print("v")
w.pdf("model_ss").fitTo(data_ss, ROOT.RooFit.PrintLevel(0)) # PrintLevel(0) suppresses printouts. Change to see things like fit results and correlation martrix

# Frame for drawing
f = w.var("mass").frame()

# Plot the data
c1 = ROOT.TCanvas()
c1.cd()
data.plotOn(f)
w.pdf("model").plotOn(f)
#TODO draw ss version on a separate plot
w.pdf("model").plotOn(f, ROOT.RooFit.Components(w.pdf("bkg")), ROOT.RooFit.LineStyle(ROOT.kDashed))
w.pdf("model_ss").plotOn(f, ROOT.RooFit.LineColor(ROOT.kRed))
w.pdf("model_ss").plotOn(f, ROOT.RooFit.Components(w.pdf("bkg_ss")), ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed))
f.Draw()
c1.SaveAs("fit_osz.png")

# Compute total signal (sum of both Gaussian terms)
nsig = w.var("nsig").getVal() + w.var("nsig1").getVal()
error = np.sqrt(np.sum(np.square([w.var("nsig").getError(), w.var("nsig1").getError()])))
print(f'Total opposite-signed signal = {nsig:.0f} +/- {error:.2f}')
nsig_ss = w.var("nsig_ss").getVal() + w.var("nsig1_ss").getVal()
error_ss = np.sqrt(np.sum(np.square([w.var("nsig_ss").getError(), w.var("nsig1_ss").getError()])))
print(f'Total same-signed signal = {nsig_ss:.0f} +/- {error_ss:.2f}')
#TODO this ratio will have to be pT and eta dependent -> add regions to the processor and/or histogram axes
print(f'Total ratio = {nsig_ss/nsig:.3f}')
