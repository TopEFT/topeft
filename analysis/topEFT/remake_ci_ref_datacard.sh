# This script reproduces the reference datacard file that the CI compares agains
# Run this script when you want to update the reference file
# Make sure you've run pytest already, to generate the files to copy

cp ../../histos/ttx_multileptons-2lss_p_2b.txt test/ttx_multileptons-2lss_p_2b_ref.txt
cp ../../histos/ttx_multileptons-2lss_p_4j_2b_ht.txt test/ttx_multileptons-2lss_p_4j_2b_ht_ref.txt
cp ../../histos/ttx_multileptons-3l_sfz_1b.txt test/ttx_multileptons-3l_sfz_1b_ref.txt
cp ../../histos/ttx_multileptons-3l_onZ_1b_2j_ptbl.txt test/ttx_multileptons-3l_onZ_1b_2j_ptbl_ref.txt
