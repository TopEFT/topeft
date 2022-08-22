Visualize Extreme Events

## Event Display

Copy the `.ig` file to local. Open the file in iSpy. 
https://ispy-webgl.web.cern.ch/

## File Information

* njets: 12 jets event
* nleps: 4 leptons event with top lepton pt (553GeV)
* pt_j: top 2 jet pt (1.8TeV, 1.6TeV), top S_T (3.7TeV), and top H_T (3.5TeV) event

## Procedure to get the `.ig` file

* Modify `find_file.py` to generate a dataframe of selected event(s) from the skimmed files.
* Modify `run_extreme_events.py` to run the find file processor (replace extreme_events with find_file in the script). 
* Run `find_file.py` on the non-skimmed data `data_samples.cfg`.
* Get run, luminosityBlock, event, and the non-skimmed root file of the event from the output dataframe.
* Use `dasgoclient` to find the AOD file:

```
dasgoclient -query parent file=/store/data/Run2017B/DoubleMuon/NANOAOD/path_of_the_root_file_from_the_previous_step.root

# Use one file from the output above
dasgoclient -query parent file=/store/data/Run2017B/DoubleMuon/MINIAOD/path_of_the_root_file_from_the_previous_step.root

dasgoclient -query dataset file=/store/data/Run2017B/DoubleMuon/AOD/path_of_the_root_file_from_the_previous_step.root

dasgoclient -query file,run,lumi dataset=/output/from/above
```

The last output would be the path of the AOD file, run, and a list of luminosityBlock. The run and one of the luminosityBlock have to match the selected event.
* Compile iSpy following the [instructions](https://github.com/cms-outreach/ispy-analyzers). For the 3 events in the folder, the matching CMSSW version is CMSSW_10_6_2.
* Use `edmCopyPickMerge` with run, luminosityBlock, event, and the AOD file from the last output to generate an event file with the format required by iSpy. An example:

```
edmCopyPickMerge outputFile=pickevents.root   eventsToProcess=297296:266:385206686   inputFiles=/store/data/Run2017B/DoubleMuon/AOD/09Aug2019_UL2017-v1/260000/323E97B3-C70A-E44A-922F-7E683021C1EB.root
```

* Run the iSpy analyzer on `pickevents.root` to generate the `.ig` file.

