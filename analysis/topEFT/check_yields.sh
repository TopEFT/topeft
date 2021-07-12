#!/bin/bash


# What we want to call the output files
FILE_NAME="check_yields"

# The json we want to compare against
REF_FILE_NAME="test/placeholder_ref_yields.json"


# Run the processor
printf "\nRunning processor...\n"
time python run.py ../../topcoffea/cfg/check_yields_sample.cfg -o $FILE_NAME

# Make the jsons
printf "\nMaking yields json from pkl...\n"
python get_yield_json.py histos/$FILE_NAME.pkl.gz -n $FILE_NAME --quiet

# If we want this to be the new ref json (do we want to have an option for this?):
#cp $FILE_NAME.json tests/$REF_FILE_NAME

# Compare the yields to the ref json
printf "\nComparing yields agains reference...\n"
python comp_yields.py -f1 $REF_FILE_NAME -f2 $FILE_NAME.json -t1 "Ref yields" -t2 "New yields" --quiet

# Do something with the exit code?
echo $?
