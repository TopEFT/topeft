# This script reproduces the reference datacard file that the CI compares against
# Run this script when you want to update the reference file

# Get the file the CI uses
printf "\nDownloading root file...\n"
wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root

# Run the processor
printf "\nRunning the processor...\n"
time python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json -o new_ref_histos --do-np -x futures "$@"

# Run the datacard maker
printf "\nRunning the datacard maker...\n"
python make_cards.py histos/new_ref_histos_np.pkl.gz -d test --var-lst lj0pt --do-nuisance --ch-lst "2lss_p_4j" --selected-wcs-ref "test/selectedWCs_ref_ci.json"

cp test/selectedWCs.txt test/selectedWCs_ref_ci.json
