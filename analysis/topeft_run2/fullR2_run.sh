# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
COMMIT="b225975a"
TAG="run2taus"

# Build the run command for filling SR histos
#TAG="${TAG}_SRs"
#OUT_NAME="${TAG}_${COMMIT}"
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME --tau_h_analysis" # For analysis

# Build the run command for filling CR histos
TAG="${TAG}_SRs"
OUT_NAME="${TAG}_${COMMIT}"
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_cr_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
CFGS="../../input_samples/cfgs/mc_background_samples_NDSkim_loc.cfg"
OPTIONS="--hist-list cr --skip-sr --do-systs --wc-list ctG -o $OUT_NAME --tau_h_analysis --split-lep-flavor -x futures -c 1 -s 50" # For CR plots


echo "OUT_NAME:" $OUT_NAME

# Run the processor over all Run2 samples
RUN_COMMAND="time python run_analysis.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
