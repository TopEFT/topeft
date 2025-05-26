# Name the output
 
OUT_NAME="2022_btag_efficiency_map_srbg_DeepJet"
CFGS="../../input_samples/cfgs/ND_2022_signal_samples.cfg,../../input_samples/cfgs/ND_2022_background_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2023_signal_samples.cfg,../../input_samples/cfgs/ND_2023_background_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2022EE_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2022EE_signal_samples.cfg,../../input_samples/cfgs/ND_2022EE_background_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2023BPix_signal_samples.cfg,../../input_samples/cfgs/ND_2023BPix_background_samples.cfg"

OPTIONS=" -o $OUT_NAME" # add --test to run over a few set of events                                               
RUN_COMMAND="time python run.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
