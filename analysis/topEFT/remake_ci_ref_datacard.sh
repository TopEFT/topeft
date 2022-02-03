# This script reproduces the reference datacard file that the CI compares against
# Run this script when you want to update the reference file

if [[ "$PWD" == *"analysis/topEFT"* ]]; then
    cwd=$PWD
    cd ../../
fi
echo -e "Running TopCoffea first.\nMake sure you have the nanoAOD test file.\n"
wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194.root
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_topcoffea()'
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_nonprompt()'
echo -e "Running initial scans\nIgnore any errors.\n"
mkdir -p histos
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_datacard_2l()'
cp histos/ttx_multileptons-2lss_p_2b.txt analysis/topEFT/test/ttx_multileptons-2lss_p_2b_ref.txt
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_datacard_2l_ht()'
cp histos/ttx_multileptons-2lss_p_4j_2b_ht.txt analysis/topEFT/test/ttx_multileptons-2lss_p_4j_2b_ht_ref.txt
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_datacard_3l()'
cp histos/ttx_multileptons-3l_sfz_1b.txt analysis/topEFT/test/ttx_multileptons-3l_sfz_1b_ref.txt
python -c 'import tests.test_topcoffea; tests.test_topcoffea.test_datacard_3l_ptbl()'
cp histos/ttx_multileptons-3l_onZ_1b_2j_ptbl.txt analysis/topEFT/test/ttx_multileptons-3l_onZ_1b_2j_ptbl_ref.txt
echo -e "\nRunning once more to make sure everyting worked\nErrors ARE important!\n"
python analysis/topEFT/remake_ci_ref_datacard.py
cd $cwd
