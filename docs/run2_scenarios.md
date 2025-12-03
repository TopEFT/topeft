# Run‑2 scenarios, groups, and workflows

Run‑2 executions are driven entirely by metadata rather than ad‑hoc JSON lists.
Two layers work together:

* **Groups** – Logical bundles of channels defined in
  `analysis/metadata/metadata_run2_groups.yaml`. Each group enumerates the
  regions, applications, and histogram selections that the processor should
  evaluate. The `ChannelMetadataHelper` from `topeft.modules.channel_metadata`
  reads this file and exposes the groups by name.
* **Scenarios** – Named combinations of one or more groups. The mapping lives in
  `analysis/metadata/run2_scenarios.yaml` and is loaded by
  `analysis/topeft_run2/run2_scenarios.py`. The CLI and quickstart helpers look
  up scenarios through `analysis/topeft_run2/scenario_registry.py`, which
  returns the correct metadata bundle (for example `metadata_TOP_22_006.yaml`)
  and tracks the names exposed to `--scenario` and YAML profiles.

When `run_analysis.py` or `full_run.sh` receives a `--scenario` value, the
workflow:

1. Resolves the scenario through `scenario_registry.py` to find the metadata
   document.
2. Loads the groups referenced in `run2_scenarios.yaml`.
3. Passes the fully expanded channel list to `ChannelMetadataHelper`, which
   feeds `ChannelPlanner` and `HistogramPlanner`.

Legacy JSON manifests (such as `scratch/ch_lst_step5.json`) are now used only by
the validation scripts; production code is 100% YAML‑driven.

## Canonical Run‑2 scenarios

| Scenario | Description | Typical uses |
| --- | --- | --- |
| `TOP_22_006` | Baseline TOP‑22‑006 reinterpretation groups plus shared control regions. | Default Run‑2 SR/CR campaigns, QA launches, and `fullR2_run.yml` presets. |
| `tau_analysis` | Adds tau‑enriched SRs/CRs on top of the core control suite. | Tau reinterpretation studies and datacard cross‑checks. |
| `offz_tau_analysis` | Keeps the off‑Z split used during tau studies without activating the forward categories. | Targeted comparisons against the tau analysis without the full scenario stack. |
| `offz_fwd_analysis` | Preserves the forward/off‑Z split required for forward‑jet studies. | Forward‑jet reinterpretations and closure tests. |
| `all_analysis` | Union of the canonical Run‑2 groups (core, tau, forward, off‑Z). | Comprehensive bookkeeping runs or datacard production that needs every category at once. |

The Run‑3 default scenario is `fwd_analysis` (reused from the table above), but
when you request Run‑3 eras the wrapper automatically switches to the Run‑3
metadata bundle.

## Combining scenarios

Scenarios can be combined on the CLI or inside YAML:

```bash
python analysis/topeft_run2/run_analysis.py \
    input_samples/cfgs/mc_signal_samples_NDSkim.cfg \
    --scenario TOP_22_006 \
    --scenario tau_analysis \
    --scenario offz_fwd_analysis \
    --executor iterative --nchunks 1 --chunksize 5
```

When you supply `--options path.yml` the YAML becomes authoritative. Encode the
scenario list there instead of repeating `--scenario`:

```yaml
# analysis/topeft_run2/configs/fullR2_run_tau_fwd.yml
defaults:
  scenarios:
    - TOP_22_006
    - tau_analysis
    - offz_fwd_analysis
profiles:
  sr:
    skip_cr: true
  cr:
    skip_sr: true
```

Remember that the CLI enforces a mutual exclusion rule: `--scenario` cannot be
combined with `--options`. The wrapper (`full_run.sh`) surfaces the same guard
for convenience.

## Adding or modifying a scenario

1. **Update the metadata groups** – Extend
   `analysis/metadata/metadata_run2_groups.yaml` (or the relevant metadata file)
   with the new regions, variables, or applications. Run
   `python -m topeft.quickstart --emit-options` to sanity‑check your changes.
2. **Register the group composition** – Edit
   `analysis/metadata/run2_scenarios.yaml` to add or modify the scenario name
   and its group list. Keep names descriptive and avoid overlapping the
   canonical identifiers above unless you intend to redefine them.
3. **Expose the scenario to the CLI** – If the metadata bundle changed, update
   `analysis/topeft_run2/scenario_registry.py` so the new metadata path is
   discoverable via `--scenario`. For Run‑2 additions this typically means
   reusing `metadata_run2_groups.yaml` and pointing the registry to the same
   file.
4. **Validate** – Run the Step‑5 validators from `scratch/` to confirm the new
   scenario is internally consistent:

   ```bash
   PYTHONNOUSERSITE=1 PYTHONPATH="" /users/apiccine/work/miniconda3/envs/coffea2025/bin/python \
       scratch/validate_run2_scenarios_step5.py
   ```

   The scripts compare the YAML source against the generated channel lists and
   will raise if group definitions conflict.

By following these steps the CLI, quickstart helpers, and `full_run.sh` wrapper
will all recognize the scenario automatically, keeping the Run‑2 documentation
and workflow in sync with the metadata source of truth.
