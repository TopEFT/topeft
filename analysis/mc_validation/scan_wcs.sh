#!/usr/bin/env bash
wcs=("cHQ1" "cHQ3" "cHt" "cHtbRe" "cleQt1Re11" "cleQt1Re22" "cleQt1Re33" "cleQt3Re11" "cleQt3Re22" "cleQt3Re33" "cQe1" "cQe2" "cQe3" "cQj11" "cQj18" "cQj31" "cQj38" "cQl11" "cQl12" "cQl31" "cQl32" "cQl33" "ctBRe" "cte1" "cte2" "cte3" "ctGRe" "ctHRe" "ctj1" "ctj8" "ctl1" "ctl2" "ctWRe")
proc="ttll"

wcs=("cQu8" "cte1" "cQu1" "ctt" "cQd8" "clu" "ctWRe" "cHt" "cHQ1" "cbBRe" "cte3" "cte2" "ctHRe" "cleQt1Re11" "cQj31" "cQe3" "cbWRe" "cHtbRe" "cleQt1Re33" "cQl32" "cQj11" "cQb8" "cQe1" "cQt1" "cQd1" "cQl31" "cHbox" "cleQt1Re22" "cQj38" "ctu8" "cQl33" "ctGRe" "ctj1" "cQl11" "cQe2" "ctb8" "ctl2" "ctl1" "cQl12" "ctj8" "ctu1" "cQj18" "cleQt3Re22" "ctd8" "cQt8" "cleQt3Re33" "clj1" "cQQ1" "cld" "cHQ3")
proc="tttt"


wsc=("cbBRe" "cbWRe" "cHbox" "cHQ1" "cHQ3" "cHt" "cHtbRe" "cld" "cleQt1Re11" "cleQt1Re22" "cleQt1Re33" "cleQt3Re22" "cleQt3Re33" "clj1" "clu" "cQb8" "cQd1" "cQd8" "cQe1" "cQe2" "cQe3" "cQj11" "cQj18" "cQj31" "cQj38" "cQl11" "cQl12" "cQl31" "cQl32" "cQl33" "cQQ1" "cQt1" "cQt8" "cQu1" "cQu8" "ctb8" "ctd8" "cte1" "cte2" "cte3" "ctGRe" "ctj1" "ctj8" "ctl1" "ctl2" "ctt" "ctu1" "ctu8" "ctWRe")
wsc=("cbBRe" "cbWRe" "cHbox" "cHQ1" "cHQ3" "cHt" "cHtbRe" "cld" "cleQt1Re11" "cleQt1Re22" "cleQt1Re33" "cleQt3Re22" "cleQt3Re33" "clj1" "clu" "cQb8" "cQd1" "cQd8" "cQe1" "cQe2" "cQe3" "cQj11" "cQj18" "cQj31" "cQj38" "cQl11" "cQl12" "cQl31" "cQl32" "cQl33" "cQu1" "cQu8" "ctb8" "ctd8" "cte1" "cte2" "cte3" "ctGRe" "ctj1" "ctj8" "ctl1" "ctl2" "ctu1" "ctu8" "ctWRe" "ctd1" "ctd8")

# Current list of SMEFTSim WCs we'd want to use
wcs=("ctHRe" "cHQ1" "ctWRe" "ctBRe" "ctGRe" "cbWRe" "cHQ3" "cHtbRe" "cHt"  "cQl31" "cQl32" "cQl33" "cQl11" "cQl12" "cQl12" "cQe1" "cQe2" "cQe3" "ctl1" "ctl2" "ctl2" "cte1" "cte2" "cte3"  "cleQt3Re11"  "cleQt3Re22"  "cleQt3Re33"  "cleQt1Re11"  "cleQt1Re22"  "cleQt1Re33"  "cQj31" "cQj38" "cQj11" "ctj1" "cQj18" "ctj8"  "clj1"  "cHbox"  "ctu1"  "ctb8"  "clu"  "cld"  "cQb8"  "ctd8"  "cQd1"  "cQd8"  "ctd1"  "cQu1"  "cbBRe"  "ctu8"  "cQu8")

four_heavy=("ctt" "cQQ1" "cQt1" "cQt8")


# Current list of dim6top WCs used in TOP-22-006
wcs=("ctW" "ctZ" "ctp" "cpQM" "ctG" "cbW" "cpQ3" "cptb" "cpt" "cQl3i" "cQlMi" "cQei" "ctli" "ctei" "ctlSi" "ctlTi" "cQq13" "cQq83" "cQq11" "ctq1" "cQq81" "ctq8" "ctt1" "cQQ1" "cQt8" "cQt1")

four_heavy=("ctt1" "cQQ1" "cQt1" "cQt8")

# Current list of dim6top WCs including new WCs not in TOP-22-006 that will be in the Run 2 + Run 3 analysis
wcs=("ctW" "ctZ" "ctp" "cpQM" "ctG" "cbW" "cpQ3" "cptb" "cpt" "cQl3i" "cQlMi" "cQei" "ctli" "ctei" "ctlSi" "ctlTi" "cQq13" "cQq83" "cQq11" "ctq1" "cQq81" "ctq8" "ctt1" "cQQ1" "cQt8" "cQt1" "ctu1" "ctb8" "cQb8" "ctd8" "cQd1" "cQd8" "ctd1" "cQu1" "ctu8" "cQu8")

four_heavy=("ctt1" "cQQ1" "cQt1" "cQt8")

if [ -n "$1" ]; then
    proc=$1
fi

if [[ "tttt" == *"$proc"* ]]; then
    wcs=("${wcs[@]}" "${four_heavy}")
    echo "Parsing tttt, adding four heavy qurak WCs"
fi

tag="Run3Dim6TopWithTOP22006AxisScan"
wc_tag="7pts_500"

rm log_wcs_${proc}
for wc in ${wcs[@]}
do
    #python validate_eft_wc.py --tolerance 3 --wc $wc --proc $proc --tag Run3With52WCsSMEFTsimTopMasslessAxisScan 2>&1 | tee -a log_wcs_${proc}
    #python validate_eft_wc.py --tolerance 3 --wc $wc --proc $proc --tag Run3With52WCsSMEFTsimTopMasslessTOP22006AxisScan --wc-tag 7pts 2>&1 | tee -a log_wcs_${proc}
    python validate_eft_wc.py --tolerance 3 --wc $wc --proc $proc --tag $tag --wc-tag $wc_tag 2>&1 | tee -a log_wcs_${proc}
done
