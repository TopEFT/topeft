# Workflow and YAML overview

> **Start here:** this hub is the recommended entry point for new contributors.
> It links the shared environment, YAML presets, executor choices, and quickstart
> workflows so you can go from “fresh checkout” to “first Run‑2 launch” without
> jumping across multiple README files. For a fuller documentation map, see
> [docs/index.md](index.md).

This hub pulls together the essentials for launching Run 2 analyses, choosing between the local futures and distributed TaskVine executors, and understanding how the YAML options files drive the workflow. It is meant as a single place to start, with pointers back to the detailed references once you are comfortable with the flow.

## Before you begin

1. Create or update the shared `coffea2025` Conda environment and install both
   `topeft` and the sibling [`topcoffea`](https://github.com/TopEFT/topcoffea)
   checkout in editable mode. The commands are summarised in the repository
   `README.md` and expanded in the [environment packaging guide](environment_packaging.md).
2. Keep `topcoffea` on the `ch_update_calcoffea` branch (or matching release tag)
   so the cache-free jet/MET corrections match what the Run‑2 workflow expects.
3. If you plan to run distributed jobs, build the TaskVine environment tarball
   via `python -m topcoffea.modules.remote_environment` before launching any runs.
4. When you are ready for an end-to-end smoke test, follow the links at the end
   of this page to the Run‑2 quickstart and plotting sections.

## Quick workflow walkthrough

1. **Pick or clone an options file.** The shared presets live in `analysis/topeft_run2/configs/` (for example `fullR2_run.yml`). Each file exposes a `defaults` block and optional `profiles` so you can switch between control and signal regions or tailor the executor. See [YAML structure and merging](#yaml-structure-and-merging) for the merge order.
2. **Prepare inputs.** Point the run at one or more sample manifests such as `input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json`. Relative paths are resolved from the repository root.
3. **Run with the local futures executor** for single-node smoke tests:

   ```bash
   cd analysis/topeft_run2
   python run_analysis.py \
       ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml:sr \
       --executor futures \
       --outpath histos/local_futures \
       --outname plotsTopEFT
   ```

   The CLI flags above override any `executor` value in the YAML so you can keep a TaskVine-ready preset while still running locally.
4. **Run with the TaskVine executor** when you are ready to distribute work. Package the environment once per checkout (`python -m topcoffea.modules.remote_environment`), ensure workers point at the advertised manager name, and let the YAML supply the backend configuration:

   ```bash
   cd analysis/topeft_run2
   python run_analysis.py \
       ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml:sr \
       --executor taskvine \
       --environment-file "$(python -m topcoffea.modules.remote_environment)" \
       --outpath histos/taskvine \
       --outname plotsTopEFT
   ```

   Launch a worker pool in another terminal and pass the same tarball through `--python-env` when using TaskVine submission helpers (for example `vine_submit_workers ... --python-env "$(python -m topcoffea.modules.remote_environment)" -M ${USER}-taskvine-coffea 10`). See [TaskVine workflow quickstart](taskvine_workflow.md) for a deeper dive.
5. **Turn the histogram pickle into plots** using the existing helper once the run finishes:

   ```bash
   cd analysis/topeft_run2
   python make_cr_and_sr_plots.py \
       -f histos/taskvine/plotsTopEFT.pkl.gz \
       -o plots/taskvine \
       -n plots \
       -y 2017 \
       --skip-syst
   ```

Both executors emit tuple-keyed histogram pickles in the `(variable, channel, application, sample, systematic)` format that the plotting utilities expect. [analysis_processing.md](analysis_processing.md) describes the tuple schema in more detail. The Run 2 processor also recomputes b-tag multiplicities as integer arrays right before histogramming so Awkward arithmetic has well-defined inputs when building category masks.

## YAML structure and merging

Options files allow you to keep executor choices, region toggles, and metadata overrides in one place. The workflow consumes them in a predictable order:

- `defaults` – applied first and always active. Good for repository-wide settings such as `summary_verbosity: full`, `do_systs: true`, or a custom `metadata` file cloned from `topeft/params/metadata.yml`.
- `profiles` – named overlays that activate with a `:profile` suffix (for example `configs/fullR2_run.yml:cr`). If only one profile exists, it is applied automatically. Profiles commonly toggle `scenarios`, `regions`, or `executor` values.
- Top-level keys – any values placed alongside `defaults` or `profiles` are merged last. Use them sparingly for ad-hoc tweaks you do not want to codify in a profile.

When `--options` is provided, the YAML becomes the single source of truth: CLI flags such as `--executor` or `--summary-verbosity` are ignored unless you explicitly pass them to override the merged result. The [`run_analysis.py` CLI and YAML reference](run_analysis_cli_reference.md) lists every supported key, while the [Run analysis configuration flow](run_analysis_configuration.md) explains how the builder resolves types and validates inputs.

### Common sections to edit

- **Executor selection and resources** – set `executor: futures` for local runs or `executor: taskvine` to distribute work. TaskVine-specific keys such as `environment_file`, `manager_name_template`, and `resources_mode` can live under `defaults` or individual profiles so they travel with the preset.
- **Metadata bundle** – point `metadata` at a cloned YAML (for example `configs/metadata_custom.yml`) when testing new regions, variables, or systematics. The [Run configuration dataclasses and metadata overview](dataclasses_and_metadata.md) and [sample metadata reference](sample_metadata_reference.md) document the available fields.
- **Samples and scenarios** – use `samples` to list JSON manifests or `.cfg` bundles, and set `scenarios` to restrict which channel groups are active. Profiles can also toggle `skip_sr`, `skip_cr`, or `years` to match the intended slice of the analysis.
- **Output naming** – adjust `outname` and `outpath` to keep control- and signal-region runs separate. Because histogram pickles encode the tuple keys, you can point plotting helpers at any of these outputs without reconfiguring channel metadata.

### Suggested editing workflow

1. Copy the nearest preset (for example `cp analysis/topeft_run2/configs/fullR2_run.yml analysis/topeft_run2/configs/fullR2_run_taskvine.yml`).
2. Update `defaults.executor` to `taskvine` and add TaskVine resource hints if you want a dedicated distributed profile; keep a separate profile for local futures runs if needed.
3. Clone `topeft/params/metadata.yml` to a tracked location and reference it via the `metadata` key when testing new variables or systematics.
4. Check the merged configuration with `python run_analysis.py ... --options <file>:<profile> --summary-verbosity full --dry-run` to confirm the chosen samples, channels, and executor before launching a long run.

Keeping these edits in YAML ensures collaborators can reproduce the exact configuration without hunting through older notebooks or shell history.
