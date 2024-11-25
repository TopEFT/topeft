# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
OUT_NAME="all_split_lep_signal"

# Build the run command for filling SR histos
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME" # For analysis

# Build the run command for filling CR histos
CFGS="../../input_samples/cfgs/2022_mc_signal_samples.cfg,../../input_samples/cfgs/2022_mc_background_samples.cfg,../../input_samples/cfgs/2022_data_samples.cfg"
OPTIONS="--hist-list cr --skip-sr  -s 50000  -o $OUT_NAME -p /scratch365/aehnis/ --split-lep-flavor" # For CR plots

# Run the processor over all Run2 samples
RUN_COMMAND="time python run_analysis.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
