:warning: **NOTE**: The scripts in this directory have not been updated since the `topcoffea` refactoring (of August 2023), if they are needed in the future, likely some import statements will need to be updated.

# Extreme Events Study

The `extreme_events` processor is designed to search for top high-energy events from the multi-lepton data. The first part of the study is getting the actual data in the form of dataframes from the processor. The second part is getting the yields from the Monte Carlo data. There are two ways of getting the yields: summing up the bins of the output histogram from the `topeft` processor and summing up the columns of the output dataframe from the `extreme_events` processor. Since it is less likely to make a mistake with `topeft`, it is recommended as the first method to try.

## How to get dataframes from the output of `extreme_events.py`?

The output of the processor is a dictionary of `dataframe_accumulator` objects. The keys in the dictionary are the event characteristics (e.g. nleps is the lepton multiplicity). Dataframes are stored as `.values` of the items in the dictionary. An example of getting two dataframes:

```
import pickle
import gzip

# Load the output
with gzip.open("path/to/output/file", "rb") as infile:
    output = pickle.load(infile)

# Get dataframes by keys (e.g. nleps, pt_j)
df_nleps = output["nleps"].value
df_pt_j = output["pt_j"].value
```

## How to get the yields from the MC samples?

### 1. Use the output histogram from `topeft.py`

* When initializing the histogram, adjust the bins to include the top events with clear divisions. If the bin is not there, add the event quantity to the variable list. Then add the bin to the histogram.
* Run `topeft.py` on the MC samples for a specific group of events. Histogram contents are now
  controlled through the `histogram_variables` include/exclude lists in `params/metadata.yml`.
  Adjust those metadata entries if additional observables are required. An example command for
  top jet multiplicity events in signal samples:

```
python work_queue_run.py ../../topcoffea/cfg/mc_signal_samples_NDSkim.cfg --skip-cr --do-np
```

* Run `get_histo_yield.py` to get the yields. 

### 2. Use the output dataframe from `extreme_events.py`

* Uncomment the EFT coefficients section.
* Add `events["yield"]` to the initial dataframe as a column like other event quantities (e.g. nleps).
* After the events are further filtered by interesting characteristics, get the output dataframe (e.g. df_nleps).
* Calculate the yield by summing up the yield column (e.g. `df_nleps['yield'].sum()`).

