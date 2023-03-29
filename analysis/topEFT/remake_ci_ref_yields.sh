# This script reproduces the reference yields file that the CI compares agains
# Run this script when you want to update the reference file

# Get the file the CI uses, and move it to the directory the JSON expects
printf "\nDownloading root file...\n"
wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root

# Run the processor
printf "\nRunning the processor...\n"
time python run_topcoffea.py ../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json -x futures -o new_ref_histos

# Make the JSON file of the yields
printf "\nMaking the yields JSON file...\n"
python get_yield_json.py -f histos/new_ref_histos.pkl.gz -n new_ref_yields

# Replace the reference yields with the new reference yields
printf "\nReplacing ref yields JSON with new file...\n"
mv new_ref_yields.json test/UL17_private_ttH_for_CI_yields.json
printf "\n\nDone.\n\n"
