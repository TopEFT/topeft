# Extreme Events Study

## How to get dataframes from the output of `extreme_events.py`?

There are multiple dataframes stored as the values of the `dataframe_accumulator` objects in the `dict` output.

```
import pickle
import gzip

# Load output
with gzip.open("path/to/output/file", "rb") as infile:
    output = pickle.load(infile)

# Get dataframes by keys (e.g. nleps, pt_j)
df_nleps = output["nleps"].value
df_pt_j = output["pt_j"].value
```

## How to get the yields from the MC samples?

### 1. Use the output histogram from `topeft.py`

* When initializing the histogram, adjust the bins to include the top events with clear divisions. If the bin is not there, add the event quantity to the variable list. Then add the bin to the histogram.
* Run `topeft.py` on the MC samples for a specific group of events. An example of top jet multiplicity events in signal samples:

```
python work_queue_run.py ../../topcoffea/cfg/mc_signal_samples_NDSkim.cfg --hist-list njets --skip-cr --do-np
```

* Run `get_histo_yield.py` to get the yields. 

### 2. Use the output dataframe from `extreme_events.py`

* Uncomment the EFT coefficients section.
* Add `events["yield"]` to the initial dataframe as a column like other event quantities (e.g. nleps).
* After the events are further filtered by interesting characteristics, get the output dataframe (e.g. df_nleps).
* Calculate the yield by summing up the yield column (e.g. `df_nleps['yield'].sum()`).

