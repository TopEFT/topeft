# This script runs the wq run script with all of the settings appropriate for making SR histos for the full R2 analysis

# Name the output
OUT_NAME="flip_test"

# Build the run command for filling SR histos
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg,../../input_samples/cfgs/mc_background_samples_NDSkim.cfg,../../input_samples/cfgs/data_samples_NDSkim.cfg"
#OPTIONS="--hist-list ana --skip-cr --do-systs -s 50000 --do-np -o $OUT_NAME" # For analysis

# Build the run command for filling CR histos
#CFGS="../../input_samples/cfgs/ND_2022_signal_samples.cfg,../../input_samples/cfgs/ND_2022_background_samples.cfg,../../input_samples/cfgs/2022_data.cfg"
CFGS="../../input_samples/cfgs/NDSkim_2022_background_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2022EE_background_samples_loc.cfg"
#CFGS="../../input_samples/cfgs/ND_2023_background_samples_loc.cfg"
#CFGS="../../input_samples/cfgs/2022EE_data.cfg"
#CFGS="../../input_samples/cfgs/ND_2023BPix_background_samples_loc.cfg"
#CFGS="../../input_samples/cfgs/ND_2022_signal_samples.cfg"
OPTIONS="-x futures -p histoR3 -o $OUT_NAME " #-s 5 -c 1 " # For CR plots
# Run the processor over all Run2 samples
RUN_COMMAND="time python run_flip.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
