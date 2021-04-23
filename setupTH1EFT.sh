#!/bin/bash

echo "Downloading files"
wget -O Utils/TH1EFT.cc https://raw.githubusercontent.com/TopEFT/EFTGenReader/master/EFTHelperUtilities/src/TH1EFT.cc --quiet
wget -O Utils/WCFit.h https://raw.githubusercontent.com/TopEFT/EFTGenReader/master/EFTHelperUtilities/interface/WCFit.h --quiet
wget -O Utils/WCPoint.h https://raw.githubusercontent.com/TopEFT/EFTGenReader/master/EFTHelperUtilities/interface/WCPoint.h --quiet

echo "Compiling ROOT files (WCFit.h and TH1EFT.cc)"
{
root -q -b Utils/WCFit.h+

#Change path (Utils folder)
sed -i 's/EFT.*TH1EFT.h/TH1EFT.h/' Utils/TH1EFT.cc

root -q -b Utils/TH1EFT.cc+ > /dev/null 2>&1
} > /dev/null 2>&1
