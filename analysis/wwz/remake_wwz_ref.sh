# This script reproduces the reference yields file that the CI compares against
# Run this script when you want to update the reference file

# Get the file the CI uses, and move it to the directory the JSON expects
printf "\nDownloading root file...\n"
wget -nc http://uaf-10.t2.ucsd.edu/~kmohrman/for_ci/for_wwz/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep/output_1.root

# Run the processor
printf "\nRunning the processor...\n"
time python run_wwz4l.py ../../input_samples/sample_jsons/test_samples/UL17_WWZJetsTo4L2Nu_forCI.json -x futures -o new_ref_histos

# Make the JSON file of the yields
printf "\nMaking the yields JSON file...\n"
python get_wwz_counts.py -f histos/new_ref_histos.pkl.gz -n new_ref_yields

# Compare the JSON file of the yields
printf "\nCompare the new yields JSON file to old ref...\n"
python comp_json_yields.py new_ref_yields.json ref_for_ci/counts_wwz_ref.json -t1 "New yields" -t2 "Old ref yields"

# Replace the reference yields with the new reference yields
printf "\nReplacing ref yields JSON with new file...\n"
mv new_ref_yields.json ref_for_ci/counts_wwz_ref.json
printf "\n\nDone.\n\n"
