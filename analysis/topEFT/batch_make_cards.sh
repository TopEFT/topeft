#!/bin/bash

# Hacky way of "batchifying" the new datacard maker. Takes 1 required input argument that specifies
#   the path to the pkl file to be used. The optional 2nd argument specifies the output directory
#   to save the output files to.

if [ "$(hostname)" != "earth.crc.nd.edu" ]; then
    echo "This script is expected to be ran from earth!"
    exit 0
fi

# INF="/scratch365/awightma/datacards_TOP-22-006/may26_fullRun2_withSys_anatest10_np.pkl.gz"
INF=${1}
OUT_DIR=${2}
DIST="lj0pt"

if [ -z ${OUT_DIR} ]; then
    if [ ! -d "/scratch365/${USER}" ]; then
        echo "User does not appear to have a /scratch365 space, please explicitly specify an output directory to use"
        exit 0
    fi
    # No input directory was specified, use a default
    OUT_DIR="/scratch365/${USER}/datacards"
fi

mkdir -p ${OUT_DIR}

# POIS="cpt,cptb,cQlMi,ctG,ctW,ctZ,cQl3i,ctlTi,ctq1,ctli,cQq13,cbW,cpQM,cpQ3,ctei,cQei" # 16WC
# POIS="cpt,cptb,cQlMi,ctG,ctW,ctZ,cQl3i,ctlTi" # 8WC

# IGNORE="tttt" # 5sgnl_5bkgd
# IGNORE="tttt ttlnuJet tllq tHq ttHJet TTGamma WWW WWW_4F WWZ_4F WWZ WZZ ZZZ flips nonprompt" # 1sgnl_1bkgd

# Split the processing up into separate distinct groups of channels
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "2lss_p_.*" "2lss_m_.*" >& out_2l.log &
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "2lss_4t_.*" >& out_2l_4t.log &
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "3l_onZ_.*" >& out_3l_onZ.log &
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "3l_p_offZ_.*" >& out_3l_p_offZ.log &
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "3l_m_offZ_.*" >& out_3l_m_offZ.log &
nohup python make_cards.py ${INF} -d ${OUT_DIR} --var-lst ${DIST} --do-nuisance --ch-lst "4l_.*" >& out_4l.log &