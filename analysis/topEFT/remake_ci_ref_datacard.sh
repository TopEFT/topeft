# This script reproduces the reference datacard file that the CI compares agains
# Run this script when you want to update the reference file
# Make sure you've run pytest already, to generate the files to copy

echo -e "Running initial scans\n"
if [[ "$PWD" == *"analysis/topEFT"* ]]; then
    cwd=$PWD
    cd ../../
fi
python analysis/topEFT/remake_ci_ref_datacard.py
cp histos/ttx_multileptons-2lss_p_2b.txt analysis/topEFT/test/ttx_multileptons-2lss_p_2b_ref.txt
cp histos/ttx_multileptons-2lss_p_4j_2b_ht.txt analysis/topEFT/test/ttx_multileptons-2lss_p_4j_2b_ht_ref.txt
cp histos/ttx_multileptons-3l_sfz_1b.txt analysis/topEFT/test/ttx_multileptons-3l_sfz_1b_ref.txt
cp histos/ttx_multileptons-3l_onZ_1b_2j_ptbl.txt analysis/topEFT/test/ttx_multileptons-3l_onZ_1b_2j_ptbl_ref.txt
echo -e "\nRunning onces more to make sure everyting worked\n"
python analysis/topEFT/remake_ci_ref_datacard.py --final
cd $cwd
