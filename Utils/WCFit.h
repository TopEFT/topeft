#ifndef WCFIT_H_
#define WCFIT_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <tuple>    // for std::pair

#include "WCPoint.h"
#include "split_string.h"

#include "TMatrixD.h"
#include "TVectorD.h"
#include "TDecompSVD.h"

class WCFit
{
private:
    // Using vectors here instead of map to ensure ordering
    std::vector<std::string> names; // Note: Includes 'sm'
    std::vector<std::pair<int,int>> pairs;  // The pair dublets are the indicies of the 'names' vector
    std::vector<double> coeffs; // The fit structure constants

    std::vector<std::pair<int,int>> err_pairs;  // The pair dublets are the indicies of the 'pairs' vector
    std::vector<float> err_coeffs; // The error fit structure constants

    std::string tag;    // Names the fit, for id

    int kPad = 12;

public:
    WCFit(){
        this->tag = "";
    }

    WCFit(std::vector<WCPoint> pts,std::string _tag) {
        this->fitPoints(pts);
        this->setTag(_tag);
    }

    ~WCFit(){
        this->clear();
    }

    std::string kSMstr = "sm";    // Need to figure out how to make this a global constant...

    void setTag(std::string _tag) {
        this->tag = _tag;
    }

    // The number of pairs in the fit, should be equal to 1 + 2N + N(N-1)/2
    uint size() const {
        //Note: pairs.size() and coeffs.size() should always be in 1-to-1 correspondance!
        return this->pairs.size();
    }

    // The number of pairs in the error fit
    uint errSize() const {
        //Note: err_pairs.size() and err_coeffs.size() should always be in 1-to-1 correspondance!
        return this->err_pairs.size();
    }

    std::string getTag() const {
        return this->tag;
    }

    // A vector of all non-zero WCs in the fit (includes 'sm')
    std::vector<std::string> getNames() const {
        return this->names;
    }

    // A vector of (ordered) indicies, indicating the WC names of the pairs in the quadratic function
    std::vector<std::pair<int,int>> getPairs() const {
        return this->pairs;
    }

    // A vector of the coefficients for each term in the quadratic function
    std::vector<double> getCoefficients() const {
        return this->coeffs;
    }

    std::vector<std::pair<int,int>> getErrorPairs() const {
        return this->err_pairs;
    }

    std::vector<float> getErrorCoefficients() const {
        return this->err_coeffs;
    }

    // Returns a (ordered) pair of indicies corresponding to a particular quadratic term
    std::pair<int,int> getIndexPair(std::string n1, std::string n2) const {
        // Convention note: idx1 <= idx2 always!
        int idx1 = -1;
        int idx2 = -1;
        int which = -1;
        for (uint i = 0; i < this->names.size(); i++) {
            if (which == -1 && n1 == this->names.at(i)) {
                idx1 = i;
                which = 1;
            } else if (which == -1 && n2 == this->names.at(i)) {
                idx1 = i;
                which = 2;
            }

            if (idx1 == -1) {
                // We haven't found the first index yet!
                continue;
            }

            if (which == 1 && n2 == this->names.at(i)) {
                idx2 = i;
                break;
            } else if (which == 2 && n1 == this->names.at(i)) {
                idx2 = i;
                break;
            }
        }

        // Return the pair in descending order
        return std::make_pair(idx2,idx1);
    }

    // Overloaded function for quickly getting a specific index pair
    std::pair<int,int> getIndexPair(uint idx) const {
        if (idx >= this->size()) {
            std::cout << "[ERROR] WCFit tried to access invalid index " << idx << std::endl;
            throw;
        }
        return this->pairs.at(idx);
    }

    // Returns a particular structure constant from the fit function
    double getCoefficient(std::string n1, std::string n2) const {
        // Note: This is a very brute force method of finding the corresponding coefficient,
        //       the overloaded function method should be used whenever possible
        auto idx_pair = this->getIndexPair(n1,n2);
        if (idx_pair.first == -1 || idx_pair.second == -1) {
            // We don't have the fit parameter pair, assume 0 (i.e. SM value)
            return 0.0;
        }

        if (this->pairs.size() != this->coeffs.size()) {
            std::cout << "[ERROR] WCFit pairs and coeffs vectors dont match! (getCoefficient)" << std::endl;
            return 0.0;
        }

        for (uint i = 0; i < this->size(); i++) {
            if (this->pairs.at(i).first == idx_pair.first && this->pairs.at(i).second == idx_pair.second) {
                return this->coeffs.at(i);
            }
        }

        // This should never happen!
        std::cout << "[ERROR] WCFit unable to find WC pair! (getCoefficient)" << std::endl;
        return 0.0;
    }

    // Overloaded function for quickly getting a specific structure constant
    double getCoefficient(uint idx) const {
        if (idx >= this->size()) {
            std::cout << "[ERROR] WCFit tried to access invalid index " << idx << std::endl;
            throw;
        }
        return this->coeffs.at(idx);
    }

    // Can only access the error coefficients directly via the err_coeffs vector
    float getErrorCoefficient(uint idx) const {
        if (idx >= this->errSize()) {
            std::cout << "[ERROR] WCFit tried to access invalid index " << idx << std::endl;
            throw;
        }
        return this->err_coeffs.at(idx);
    }

    // Returns the dimensionality of the fit (i.e. the number of WCs)
    int getDim() const {
        return this->names.size() - 1;  // Exclude 'sm' term
    }

    // Checks to see if the fit includes the specified WC
    bool hasCoefficient(std::string wc_name) {
        for (auto& s: this->names) {
            if (s == wc_name) {
                return true;
            }
        }
        return false;
    }

    // Evaluate the fit at a particular WC phase space point
    double evalPoint(WCPoint* pt) {
        uint i;
        double v,x1,x2,c;
        std::string n1,n2;
        std::pair<int,int> idx_pair;
        v = 0.0;
        for (i = 0; i < this->size(); i++) {
            c = this->coeffs.at(i);
            idx_pair = this->pairs.at(i);
            n1 = this->names.at(idx_pair.first);
            n2 = this->names.at(idx_pair.second);

            x1 = (n1 == kSMstr) ? 1.0 : pt->getStrength(n1);  // Hard set SM value to 1.0
            x2 = (n2 == kSMstr) ? 1.0 : pt->getStrength(n2);  // Hard set SM value to 1.0
            v += x1*x2*c;
        }
        return v;
    }

    // Overloaded function to evaluate the fit in 1-D at a specific WC
    double evalPoint(std::string wc_name,double val) {
        WCPoint pt;
        pt.setStrength(wc_name,val);
        return this->evalPoint(&pt);
    }

    // Evaluate the error fit at a particular WC phase space point
    double evalPointError(WCPoint* pt) {
        uint i;
        double v,x1,x2,x3,x4,c;
        std::string n1,n2,n3,n4;
        std::pair<int,int> err_pair,idx_pair;
        v = 0.0;
        for (i = 0; i < this->errSize(); i++) {
            c = this->err_coeffs.at(i);
            err_pair = this->err_pairs.at(i);

            idx_pair = this->pairs.at(err_pair.first);
            n1 = this->names.at(idx_pair.first);
            n2 = this->names.at(idx_pair.second);
            idx_pair = this->pairs.at(err_pair.second);
            n3 = this->names.at(idx_pair.first);
            n4 = this->names.at(idx_pair.second);

            x1 = (n1 == kSMstr) ? 1.0 : pt->getStrength(n1);  // Hard set SM value to 1.0
            x2 = (n2 == kSMstr) ? 1.0 : pt->getStrength(n2);  // Hard set SM value to 1.0
            x3 = (n3 == kSMstr) ? 1.0 : pt->getStrength(n3);  // Hard set SM value to 1.0
            x4 = (n4 == kSMstr) ? 1.0 : pt->getStrength(n4);  // Hard set SM value to 1.0
            v += x1*x2*x3*x4*c;
        }
        return sqrt(v);
    }

    double evalPointError(std::string wc_name,double val) {
        WCPoint pt;
        pt.setStrength(wc_name,val);
        return this->evalPointError(&pt);
    }

    void addFit(const WCFit & added_fit) {
        if (added_fit.size() == 0) return;

        if (this->size() == 0) {
            // We are an empty fit, set all values to those of the added fit
            this->names = added_fit.getNames();
            this->pairs = added_fit.getPairs();
            this->coeffs = added_fit.getCoefficients();
            this->err_pairs = added_fit.getErrorPairs();
            this->err_coeffs = added_fit.getErrorCoefficients();
            this->tag = (this->getTag().size() == 0) ? added_fit.getTag() : this->getTag();
            return;
        }

        if (this->size() != added_fit.size()) {
            std::cout << "[ERROR] WCFit mismatch in pairs! (addFit), this->size(): " << this->size() << ", added_fit.size(): " << added_fit.size() << std::endl;
            return;
        } else if (this->errSize() != added_fit.errSize()) {
            std::cout << "[ERROR] WCFit mismatch in error pairs! (addFit)" << std::endl;
            return;
        }

        for (uint i = 0; i < this->errSize(); i++) {
            if (i < this->size()) {
                this->coeffs.at(i) += added_fit.getCoefficient(i);
            }
            // It is *very* important that we keep track of the err fit coeffs separately, since Sum(f^2) != (Sum(f))^2
            this->err_coeffs.at(i) += added_fit.getErrorCoefficient(i);
        }

        //for (uint i = 0; i < this->size(); i++) {
        //    this->coeffs.at(i) += added_fit.getCoefficient(i);
        //}
    }

    //NOTE: Should check that we are scaling the error fit properly...
    void scale(double _val) {
        for (uint i = 0; i < this->size(); i++) {
            this->coeffs.at(i) *= _val;
        }
        for (uint i = 0; i < this->errSize(); i++) {
            this->err_coeffs.at(i) *= _val*_val;    // scaled by square of val
        }
    }

    void clear() {
        this->names.clear();
        this->pairs.clear();
        this->coeffs.clear();
        this->err_pairs.clear();
        this->err_coeffs.clear();
    }

    // Save the fit to a text file
    void save(std::string fpath,bool append=false) {
        if (!append) {
            std::cout << "Producing fitparams table..." << std::endl;
        }

        std::stringstream ss1,ss2;  // Header,row info
        std::string n1,n2;
        std::pair<int,int> idx_pair;

        ss1 << std::setw(kPad) << "";
        ss2 << std::setw(kPad) << this->tag;
        for (uint i = 0; i < this->size(); i++) {
            idx_pair = this->pairs.at(i);
            n1 = this->names.at(idx_pair.first);
            n2 = this->names.at(idx_pair.second);

            ss1 << std::setw(kPad) << n1 +"*"+n2;
            ss2 << std::setw(kPad) << std::to_string(this->coeffs.at(i));
        }

        std::ofstream outf;
        if (append) {
            outf.open(fpath,std::ofstream::app);
        } else {
            outf.open(fpath,std::ofstream::out | std::ofstream::trunc);
            outf << ss1.str();
        }
        outf << "\n" << ss2.str();
        outf.close();

        this->dump(append);
    }

    void dump(bool append=false,uint max_cols=13,std::string WC_name="") {
        std::stringstream ss1,ss2;  // Header,row info
        std::string n1,n2;
        std::pair<int,int> idx_pair;

        ss1 << std::setw(kPad) << "";
        ss2 << std::setw(kPad) << this->tag;
        for (uint i = 0; i < this->size(); i++) {
            if (i >= max_cols) {
                ss1 << "  ...";
                ss2 << "  ...";
                break;
            }
            idx_pair = this->pairs.at(i);
            n1 = this->names.at(idx_pair.first);
            n2 = this->names.at(idx_pair.second);

            if (WC_name == ""){
                ss1 << std::setw(kPad) << n1+"*"+n2;
                ss2 << std::setw(kPad) << std::to_string(this->coeffs.at(i));
            } else if (WC_name != ""){
                //if ( (check_str.find(n1) != std::string::npos) and (check_str.find(n2) != std::string::npos) ){ // n1 and n2 is "sm" or WC_name
                if ( (n1 == WC_name or n1 == "sm") and (n2 == WC_name or n2 == "sm") ){
                    ss1 << std::setw(kPad) << n1+"*"+n2;
                    ss2 << std::setw(kPad) << std::to_string(this->coeffs.at(i));
                }
            }

        }

        if (!append) {
            std::cout << ss1.str() << std::endl;
        }
        std::cout << ss2.str() << std::endl;
    }

    // This is how we build up all the vectors which store the fit and err_fit info
    void extend(std::string new_name) {
        //Quadratic Form Convention:
        //  Dim=0: (0,0)
        //  Dim=1: (0,0) (1,0) (1,1)
        //  Dim=2: (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
        //  Dim=3: (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
        //  etc.
        //  Note: For ALL pairs --> p.first >= p.second
        if (this->hasCoefficient(new_name)) {
            std::cout << "[ERROR] Tried to extend WCFit with a name already present! (extend)" << std::endl;
            return;
        }

        int new_idx1,new_idx2,i,j;
        //int new_idx1,i;
        std::pair<int,int> idx_pair1,idx_pair2;

        this->names.push_back(new_name);
        new_idx1 = this->names.size() - 1;
        // Extend the pairs and coeffs vectors
        for (i = 0; i <= new_idx1; i++) {
            idx_pair1 = std::make_pair(new_idx1,i);
            this->pairs.push_back(idx_pair1);
            this->coeffs.push_back(0.0);   // Extending makes no assumptions about the fit coefficients
            // Extend the err_pairs and err_coeffs vectors
            new_idx2 = this->pairs.size() - 1;
            for (j = 0; j <= new_idx2; j++) {
                idx_pair2 = std::make_pair(new_idx2,j);
                this->err_pairs.push_back(idx_pair2);
                this->err_coeffs.push_back(0.0);
            }
        }
    }

    // Extract a n-Dim quadratic fit from a collection of WC phase space points
    void fitPoints(std::vector<WCPoint> pts) {
        this->clear();

        if (pts.size() == 0) {
            // No points to fit!
            return;
        }

        this->extend(kSMstr);   // The SM term is always first
        for (auto& kv: pts.at(0).inputs) { // Assumes that all WCPoints have exact same list of WC names
            this->extend(kv.first);
        }

        uint nCols,nRows,row_idx,col_idx;
        double x1,x2;
        std::string n1,n2;
        std::pair<int,int> idx_pair;

        nCols = this->size();   // Should be equal to 1 + 2*N + N*(N - 1)/2
        nRows = pts.size();

        TMatrixD A(nRows,nCols);
        TVectorD b(nRows);

        for (row_idx = 0; row_idx < nRows; row_idx++) {
            for (col_idx = 0; col_idx < nCols; col_idx++) {
                idx_pair = this->pairs.at(col_idx);
                n1 = this->names.at(idx_pair.first);
                n2 = this->names.at(idx_pair.second);
                x1 = ((n1 == kSMstr) ? 1.0 : pts.at(row_idx).inputs[n1]);  // Hard set SM value to 1.0
                x2 = ((n2 == kSMstr) ? 1.0 : pts.at(row_idx).inputs[n2]);  // Hard set SM value to 1.0

                A(row_idx,col_idx) = x1*x2;
                b(row_idx) = pts.at(row_idx).wgt;
            }
        }

        TDecompSVD svd(A);
        bool ok;
        const TVectorD c_x = svd.Solve(b,ok);    // Solve for the fit parameters
        for (uint i = 0; i < this->errSize(); i++) {
            if (i < this->size()) {
                this->coeffs.at(i) = c_x(i);
            }
            idx_pair = this->err_pairs.at(i);
            this->err_coeffs.at(i) = (idx_pair.first == idx_pair.second) ? c_x(idx_pair.first)*c_x(idx_pair.second) : 2*c_x(idx_pair.first)*c_x(idx_pair.second);
            //this->err_coeffs.at(i) = c_x(idx_pair.first)*c_x(idx_pair.second);
        }
        //for (uint i = 0; i < this->size(); i++) {
        //    this->coeffs.at(i) = c_x(i);
        //}
    }

    void deserialize(std::vector<std::string> in_names) {
        for(auto name : in_names) this->names.push_back(name);
    }

    void deserialize(std::vector<std::string> in_scoeff, std::vector<float> in_vcoeff, std::vector<float> in_ecoeff) {
        if(in_scoeff.size() != in_vcoeff.size()) {
            std::cout << "Size mismatch! Please make sure you provide equal length coefficient vectors." << std::endl;
            return;
        }
        for(size_t i = 0; i < in_scoeff.size(); i++) {
            auto tmps = in_scoeff.at(i);
            auto pos = tmps.find("*");
            auto n1 = tmps.substr(0, pos);
            auto n2 = tmps.substr(pos+1);
            auto idx_pair = getIndexPair(n1,n2);
            for (uint i = 0; i < this->size(); i++) {
                if (this->pairs.at(i).first == idx_pair.first && this->pairs.at(i).second == idx_pair.second) {
                    coeffs[i] = in_vcoeff.at(i);
                    break;
                }
            }
        }
        for(size_t i = 0; i < in_ecoeff.size(); i++) {
            err_coeffs[i] = in_ecoeff.at(i);
        }
    }
    //ClassDef(WCFit,1); // ROOT needs this here

};

// ROOT needs this here:
ClassImp(WCFit);

#endif
/* WCFIT */
