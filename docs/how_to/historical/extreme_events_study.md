# Legacy extreme-events study

> **Archival, unsupported interface.** The code remains under
> `analysis/extreme_events_study` but has not been updated since the August 2023
> `topcoffea` refactoring. Commands below describe historical intent and include
> retired entry points. They are not current TOP-26-006 production instructions.

The `extreme_events` processor searched multilepton data for the highest-energy
events and stored selected event properties in dataframes. The historical study
also derived Monte Carlo (MC) yields in two ways: by summing bins from the
`topeft` processor histogram, or by summing a yield column from the
`extreme_events` dataframe. The histogram route was preferred because it reused
the analysis processor's selection and weighting instead of duplicating that
yield logic in dataframe operations.

## Read dataframes from `extreme_events.py` output

The processor output is a dictionary of `dataframe_accumulator` objects keyed
by event property. For example, `nleps` denotes lepton multiplicity. Each
dataframe is stored in the corresponding accumulator's `.value` attribute:

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

## Recover the historical MC-yield procedures

### Use the `topeft.py` output histogram

* Define histogram bins that cover the high-energy events with explicit
  boundaries. If the observable is absent, add it to the historical variable
  list and histogram definition.
* Historically, run the analysis processor on MC samples for the event group.
  The recorded command used the now-retired `work_queue_run.py` entry point:

```
python work_queue_run.py ../../topcoffea/cfg/mc_signal_samples_NDSkim.cfg --hist-list njets --skip-cr --do-np
```

  Do not translate that command into a current run by renaming the executable.
  Current production ownership is documented in [production](../production.md).

* Run `get_histo_yield.py` to extract the histogram yields.

### Use the `extreme_events.py` output dataframe

* Uncomment the effective field theory (EFT) coefficient section.
* Add `events["yield"]` to the initial dataframe alongside the other event
  properties, such as `nleps`.
* Apply the historical event-property filters and select the corresponding
  dataframe, such as `df_nleps`.
* Sum its yield column, for example `df_nleps["yield"].sum()`.
