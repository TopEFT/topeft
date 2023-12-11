# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
#OUT_NAME="example_name"
OUT_NAME="12112023_genphotonstudies_2lOS_ttTo2L2Nu_ttgDilep_genPartFlavLeptons_v12"

# Build the run command for filling SR histos
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,
CFGS="../../input_samples/cfgs/mc_background_samples_NDSkim.cfg"  #,../../input_samples/cfgs/data_samples_NDSkim.cfg"
OPTIONS="--hist-list photon -s 50000 -x futures -o $OUT_NAME" # For analysis

# Build the run command for filling CR histos
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_cr_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list cr --skip-sr --do-systs --do-np --wc-list ctG -o $OUT_NAME" # For CR plots

# Run the processor over all Run2 samples
RUN_COMMAND="time python run_analysis_genPhoton.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
