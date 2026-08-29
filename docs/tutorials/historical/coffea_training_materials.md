# Historical Coffea training materials

> **Archival material:** These 2021–2023 examples predate the current
> TOP-26-006 workflow and include deprecated `coffea.hist` interfaces. They are
> retained as historical learning material, not as current installation or
> analysis instructions. Start with the current
> [analysis workflow tutorial](../analysis_workflow.md).

The accompanying example scripts remain under `analysis/training/`.

[Open the historical Jupyter Notebook in Binder](https://mybinder.org/v2/gh/TopEFT/topeft/master?labpath=analysis%2Ftraining%2Fintro_coffea.hist.ipynb).

## Past tutorials

* Jun 2021: [2021 TopCoffea tutorial](https://indico.cern.ch/event/1047567/)
* Aug 2022: [2022 TopCoffea tutorial Session 1](https://indico.cern.ch/event/1188768/)
* Sep 2022: [2022 TopCoffea tutorial Session 2](https://indico.cern.ch/event/1189721/)
* Jan 2023: [2023 Advanced TopCoffea tutorial](https://indico.cern.ch/event/1228170/)

## Historical example scripts

`simple_processor.py` and `simple_run.py` form a minimal historical topcoffea
processor example. Download the example ROOT file into `analysis/training/`:

```bash
wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root
```

Then run the processor with the historical JSON path:

```bash
python simple_run.py ../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json
```

`intro_coffea.hist.py` and `intro_coffea.hist.ipynb` introduce deprecated
`coffea.hist` methods for filling, transforming, and plotting histograms. Open
the Jupyter Notebook through Coffea Casa if that historical environment is
needed. See the
[Coffea Casa access guide](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#access)
and [Git instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#using-git).
