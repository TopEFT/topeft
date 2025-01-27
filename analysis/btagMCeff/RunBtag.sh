# Name the output                                                                                                                   
OUT_NAME="btag_test"
CFGS="../../input_samples/cfgs/2022_mc_background_samples.cfg"
#CFGS="../../input_samples/cfgs/mc_signal_samples_NDSkim.cfg"
OPTIONS=" -o $OUT_NAME" # add --test to run over a few set of events                                                               

RUN_COMMAND="time python run.py $CFGS $OPTIONS"
printf "\nRunning the following command:\n$RUN_COMMAND\n\n"
$RUN_COMMAND
