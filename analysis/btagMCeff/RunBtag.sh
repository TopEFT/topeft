'''
Run2 btag eff maps are made with ttH central samples. 
Run3 btag eff maps are made with full set of central samples.
Make sure to select the right json in the cfg files.
'''

# Name the output
 
OUT_NAME="2022_btag_efficiency_map_srbg_DeepJet"

## Central samples
CFGS="../../input_samples/cfgs/NDSkim_2022_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/NDSkim_2022EE_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/NDSkim_2023_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/NDSkim_2023BPix_signal_samples.cfg"

## Private samples
#CFGS="../../input_samples/cfgs/ND_2022_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2022EE_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2023_signal_samples.cfg"
#CFGS="../../input_samples/cfgs/ND_2023BPix_signal_samples.cfg"

OPTIONS=" -o $OUT_NAME" # add --test to run over a few set of events                                               
RUN_COMMAND="time python run.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
