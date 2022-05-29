# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
OUT_NAME="example_name"

# Build the run command
CFGS="../../topcoffea/cfg/mc_signal_samples_NDSkim.cfg,../../topcoffea/cfg/mc_background_samples_NDSkim.cfg,../../topcoffea/cfg/data_samples_NDSkim.cfg"
OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME" # For analysis
#OPTIONS="--hist-list cr --skip-sr --do-systs --do-np --wc-list ctG -o $OUT_NAME" # For CR plots
RUN_COMMAND="time python work_queue_run.py $CFGS $OPTIONS"

# Run the processor over all Run2 samples
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
