# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
OUT_NAME="refact_250901_c1"

# Build the run command for filling SR histos
OUT_NAME+="_SRs"
CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg" #,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
OPTIONS="--hist-list ana --skip-cr --do-systs -o $OUT_NAME -c 1 -s 5 -x futures " # Add "--scenario tau_analysis" or "--channel-feature requires_tau" for tau studies


# Build the run command for filling CR histos
#OUT_NAME+="_CRs"
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_cr_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list cr --skip-sr --do-systs --do-np --wc-list ctG -o $OUT_NAME --split-lep-flavor" # For CR plots

echo "OUT_NAME:" $OUT_NAME

# Run the processor over all Run2 samples
RUN_COMMAND="time python run_analysis.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
