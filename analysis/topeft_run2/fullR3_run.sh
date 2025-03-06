# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
YEAR="2022"
#YEAR="2022EE"
#YEAR="2023"
#YEAR="2023BPix"
OUT_NAME="$YEAR_CRs"

# Build the run command for filling SR histos
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME" # For analysis

# Build the run command for filling CR histos
CFGS="../../input_samples/cfgs/ND_${YEAR}_background_samples.cfg,../../input_samples/cfgs/${YEAR}_data.cfg" #,../../input_samples/cfgs/ND_${YEAR}_signal_samples.cfg"
OPTIONS="--hist-list cr --skip-sr  -s 50000 --split-lep-flavor -x futures -p /scratch365/$USER/ -o $OUT_NAME " # For CR plots
# Run the processor over all Run2 samples
RUN_COMMAND="time python run_analysis.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
#$RUN_COMMAND
