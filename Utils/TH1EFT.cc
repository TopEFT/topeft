#include "TH1EFT.h"
ClassImp(TH1EFT) //-- Needed to include custom class within ROOT

TH1EFT::TH1EFT() {}
TH1EFT::~TH1EFT() {}


TH1EFT::TH1EFT(const char *name, const char *title, Int_t nbinsx, Double_t xlow, Double_t xup)
 : TH1D (name, title, nbinsx, xlow, xup) 
{
    // Create/Initialize a fit function for each bin in the histogram
    WCFit new_fit;
    for (Int_t i = 0; i < nbinsx; i++) {
        this->hist_fits.push_back(new_fit);
    }
}
void TH1EFT::SetBins(Int_t nx, Double_t xmin, Double_t xmax)
{
    // Use this function with care! Non-over/underflow bins are simply
    // erased and hist_fits re-sized with empty fits.
    
    hist_fits.clear();
    WCFit new_fit;
    for (Int_t i = 0; i < nx; i++) {
        this->hist_fits.push_back(new_fit);
    }
    
    TH1::SetBins(nx, xmin, xmax);
}

// Note: Since Clone calls Copy, this should make Clone work as well
void TH1EFT::Copy(TObject &obj) const
{
    TH1::Copy(obj);
    for (unsigned int i = 0; i < this->hist_fits.size(); i++) {
        WCFit bin_fit;
        bin_fit.addFit(this->hist_fits.at(i));
        ((TH1EFT&)obj).hist_fits.push_back(bin_fit);
    }
    WCFit of_fit;
    WCFit uf_fit;
    of_fit.addFit(this->overflow_fit);
    uf_fit.addFit(this->underflow_fit);
    ((TH1EFT&)obj).overflow_fit = of_fit;
    ((TH1EFT&)obj).underflow_fit = uf_fit;
}

Bool_t TH1EFT::Add(const TH1 *h1, Double_t c1)
{
    // check whether the object pointed to inherits from (or is a) TH1EFT:
    if (h1->IsA()->InheritsFrom(TH1EFT::Class())) {
        if (this->hist_fits.size() == ((TH1EFT*)h1)->hist_fits.size()) {
            for (unsigned int i = 0; i < this->hist_fits.size(); i++) {
                // assumes this hist and the one whose fits we're adding have the same bins!
                this->hist_fits[i].addFit( ((TH1EFT*)h1)->hist_fits[i] );
            }
        } else { 
            std::cout << "Attempt to add 2 TH1EFTs with different # of fits!" << std::endl;
            std::cout << this->hist_fits.size() << ", " << ((TH1EFT*)h1)->hist_fits.size() << std::endl;
        }
        this->overflow_fit.addFit( ((TH1EFT*)h1)->overflow_fit );
        this->underflow_fit.addFit( ((TH1EFT*)h1)->underflow_fit );
    }
    
    return TH1::Add(h1,c1); // I think this should work
}

// Custom merge function for using hadd
Long64_t TH1EFT::Merge(TCollection* list)
{
    TIter nexthist(list);
    TH1EFT *hist;
    while ((hist = (TH1EFT*)nexthist.Next())) {
        if (this->hist_fits.size() != hist->hist_fits.size()) {
            std::cout << "[WARNING] Skipping histogram with different # of fits" << std::endl;
            continue;
        }
        for (unsigned int i = 0; i < this->hist_fits.size(); i++) {
            this->hist_fits.at(i).addFit(hist->hist_fits.at(i));
        }
        this->overflow_fit.addFit(hist->overflow_fit);
        this->underflow_fit.addFit(hist->underflow_fit);
    }

    return TH1::Merge(list);
}

Int_t TH1EFT::Fill(Double_t x, Double_t w, WCFit fit)
{
    Int_t bin_idx = this->FindFixBin(x) - 1;
    Int_t nhists  = this->hist_fits.size();
    if (bin_idx >= nhists) {
        // For now ignore events which enter overflow bin
        this->overflow_fit.addFit(fit);
        return Fill(x,w);
    } else if (bin_idx < 0) {
        // For now ignore events which enter underflow bin
        this->underflow_fit.addFit(fit);
        return Fill(x,w);
    }
    this->hist_fits.at(bin_idx).addFit(fit);
    return Fill(x,w); // the original TH1D member function
}

// Returns a fit function for a particular bin (no checks are made if the bin is an over/underflow bin)
WCFit TH1EFT::GetBinFit(Int_t bin)
{
    Int_t nhists = this->hist_fits.size();
    if (bin <= 0) {
        return this->underflow_fit;
    } else if (bin > nhists) {
        return this->overflow_fit;
    }
    return this->hist_fits.at(bin - 1);
}

// Returns a WCFit whose structure constants are determined by summing structure constants from all bins
WCFit TH1EFT::GetSumFit()
{
    WCFit summed_fit;
    for (unsigned int i = 0; i < this->hist_fits.size(); i++) {
        summed_fit.addFit(this->hist_fits.at(i));
    }
    return summed_fit;
}

// Returns a bin scaled by the the corresponding fit evaluated at a particular WC point
Double_t TH1EFT::GetBinContent(Int_t bin, WCPoint wc_pt)
{
    if (this->GetBinFit(bin).getDim() <= 0) {
        // We don't have a fit for this bin, return regular bin contents
        return GetBinContent(bin);
    }

    double scale_value = this->GetBinFit(bin).evalPoint(&wc_pt);
    Double_t num_events = GetBinContent(bin);
    if (num_events == 0) {
        return 0.0;
    }

    return scale_value;
}
void TH1EFT::Scale(WCPoint wc_pt)
{
    // Warning: calling GetEntries after a call to this function will return a 
    // non-zero value, even if the histogram was never filled.
    
    for (Int_t i = 1; i <= this->GetNbinsX(); i++) {
        Double_t new_content = this->GetBinContent(i,wc_pt);
        Double_t new_error = (GetBinFit(i)).evalPointError(&wc_pt);
        this->SetBinContent(i,new_content);
        this->SetBinError(i,new_error);
    }
    
}
//// evalPointError disabled:
//void TH1EFT::Scale(WCPoint wc_pt)
//{
//    // Warning: calling GetEntries after a call to this function will return a 
//    // non-zero value, even if the histogram was never filled.
//    
//    for (Int_t i = 1; i <= this->GetNbinsX(); i++) {
//        Double_t old_content = this->GetBinContent(i);
//        Double_t new_content = this->GetBinContent(i,wc_pt);
//        Double_t old_error = this->GetBinError(i);
//        this->SetBinContent(i,new_content);
//        this->SetBinError(i,old_error*new_content/old_content);
//    }
//    
//}
// Uniformly scale all fits by amt
void TH1EFT::ScaleFits(double amt)
{
    for (uint i = 0; i < this->hist_fits.size(); i++) {
        this->hist_fits.at(i).scale(amt);
    }
}

// Display the fit parameters for all bins
void TH1EFT::DumpFits()
{
    for (uint i = 0; i < this->hist_fits.size(); i++) {
        this->hist_fits.at(i).dump();
    }
}
void TH1EFT::SetBinFit(Int_t bin, WCFit fit) {
    Int_t nhists  = this->hist_fits.size();
    if(bin == 0) underflow_fit = fit;
    else if (bin >= nhists) overflow_fit = fit;
    else hist_fits[bin-1] = fit;
}
