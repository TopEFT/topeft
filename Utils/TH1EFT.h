#ifndef TH1EFT_H_
#define TH1EFT_H_

#include "TH1D.h"
#include <vector>
#include "TClass.h"
#include "WCFit.h"
#include "WCPoint.h"

class TH1EFT : public TH1D
{
    public:
    
        // ROOT needs these:
        TH1EFT();
        ~TH1EFT();
        
        // usual constructor:
        TH1EFT(const char *name, const char *title, Int_t nbinsx, Double_t xlow, Double_t xup);
        
        std::vector<WCFit> hist_fits;
        //TODO(maybe?): Add over/underflow bin fit functions and update Fill to use them accordingly
        WCFit overflow_fit;
        WCFit underflow_fit;

        using TH1D::Fill;           // Bring the TH1D Fill fcts into scope
        using TH1D::GetBinContent;  // Bring the TH1D GetBinContent fcts into scope
        using TH1D::Scale;          // Bring the TH1D Scale fcts into scope (likely not needed)

        void Copy(TObject &obj) const;  // This allows Clone to properly copy the WCFit objects

        Int_t Fill(Double_t x, Double_t w, WCFit fit);
        WCFit GetBinFit(Int_t bin);
        WCFit GetSumFit();
        Double_t GetBinContent(Int_t bin, WCPoint wc_pt);
        //TH1EFT* Scale(WCPoint wc_pt);
        void Scale(WCPoint wc_pt);
        void ScaleFits(double amt);
        void DumpFits();
        void SetBinFit(Int_t bin, WCFit fit);
        
        void SetBins (Int_t nx, Double_t xmin, Double_t xmax);  // overriding virtual function from TH1
        Bool_t Add(const TH1 *h1, Double_t c1=1); // overriding virtual function from TH1
        Long64_t Merge(TCollection* list);

        ClassDef(TH1EFT,1); // ROOT needs this here
        //TODO(maybe?): Add member function to return specifically fit coeffs (rather then entire WCFit object)
};

// ROOT needs this here:
ClassImp(TH1EFT);

#endif
