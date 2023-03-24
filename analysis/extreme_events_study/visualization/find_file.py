#!/usr/bin/env python
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
from coffea.processor import AccumulatorABC

import pickle
import gzip

class dataframe_accumulator(AccumulatorABC):

    def __init__(self, base):
        self._base = base
        self._value = pd.DataFrame()

    @property
    def value(self):
        return self._value

    def identity(self):
        return dataframe_accumulator(pd.DataFrame())

    # Compare event information and leave only the selected events
    def add(self, other):
        if isinstance(other, pd.core.frame.DataFrame):
            df = other.merge(self._base, on=["run", "luminosityBlock", "event"])
            df = df.loc[:,~df.columns.duplicated()]
            self._value = pd.concat([self._value, df])
        else:
            df = other._value.merge(self._base, on=["run", "luminosityBlock", "event"])
            df = df.loc[:,~df.columns.duplicated()]
            self._value = pd.concat([self._value, df])
    

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        with gzip.open("dataframe/flipTopEFT.pkl.gz", "rb") as infile:
            output = pickle.load(infile)

        ############# Specify events to track #############

        # Put event information of selected events from SKIM files into one dataframe 
        df_pt_j = output["pt_j"].value[["run", "luminosityBlock", "event"]][:1]
        df_njets = output["njets"].value[["run", "luminosityBlock", "event"]][:1]
        df_nleps = output["nleps"].value.sort_values(by="pt_l_0", ascending=False)
        df_nleps = df_nleps[["run", "luminosityBlock", "event"]][:1]
        df_SKIM = pd.concat([df_pt_j, df_njets, df_nleps], ignore_index=True)

        self._accumulator = processor.dict_accumulator({
                                "nonSKIM": dataframe_accumulator(df_SKIM)
                            })

    @property
    def accumulator(self):
        return self._accumulator

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        dataset = events.metadata["dataset"]
        filename = events.metadata["filename"]

        events_info = {}
        info = ["run", "luminosityBlock", "event"]
        for label in info:
            events_info[label] = events[label]

        # Put nonSKIM file names to the output dataframe
        df = pd.DataFrame(events_info, columns=info)
        df.insert(0, "json_name", [dataset for x in range(len(df.index))])
        df.insert(1, "root_name", [filename for x in range(len(df.index))])

        self.accumulator["nonSKIM"].add(df)

        return self.accumulator

    def postprocess(self, accumulator):
        pass
