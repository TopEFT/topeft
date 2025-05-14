#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2023skim_CRs.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2023 -s -n testae23_s
#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2023skim_CRs_np.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2023 -s -n testae23_snp

#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2023BPix_CRs.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2023BPix -s -n testae23BPix
#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2023BPixskim_CRs.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2023BPix -s -n testae23BPix_s
#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2023BPixskim_CRs_np.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2023BPix -s -n testae23BPix_snp

#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2022EEskim_CRs.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2022EE -s -n testae22EE_s
#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2022EEskim_CRs_np.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2022EE -s -n testae22EE_snp

#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2022skim_CRs.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2022 -s -n testae22_s
#python make_cr_and_sr_plots.py -f /scratch365/aehnis/2022skim_CRs_np.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2022 -s -n testae22_snp


YEAR="2022"
TYPE="" #Central"
COMMIT="8f51a9b0+6b82d23f-bis"
#python make_cr_and_sr_plots.py -f histoR3/2022_run3_object_selection_test.pkl.gz -o /scratch365/apiccine/run3plots -t -y 2022 -s -n mytest22
python make_cr_and_sr_plots.py -f /scratch365/apiccine/${YEAR}CRs${TYPE}_${COMMIT}_np.pkl.gz -o /scratch365/apiccine/run3plots -t -y $YEAR -s -n test${YEAR}CRs${TYPE}_${COMMIT}
