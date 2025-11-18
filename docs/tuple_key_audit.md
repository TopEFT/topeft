# Tuple key audit

This audit records where histogram tuple keys are created and consumed across the repository. The target convention is `(variable, channel, application, sample, systematic)`; tuple consumers now reject legacy 4-tuples to avoid silent reshaping.

## Run 2 workflow
- `analysis/topeft_run2/analysis_processor.py` validates that `hist_keys` entries are 5-element tuples and builds keys as `(variable, channel, application, sample, systematic)`.【F:analysis/topeft_run2/analysis_processor.py†L322-L552】
- Downstream `HistogramCombination` usage in the Run 2 workflow therefore aligns with the 5-tuple standard.

## Training tutorial workflow
- The training processor now emits 5-tuples `(variable, channel, application, sample, systematic)` with an explicit default application tag.【F:analysis/training/simple_processor.py†L308-L379】
- Tests enforce the 5-tuple shape and require no categorical axes in the stored histograms.【F:tests/test_training_tuple_output.py†L94-L117】
- The training runner already serialises 5-tuple histogram keys for downstream consumption.【F:analysis/training/simple_run.py†L1-L117】

## Shared runner output helpers
- `topeft/modules/runner_output.py` rejects non-5-tuple histogram keys and raises on categorical histogram axes to avoid fallback reconstruction.【F:topeft/modules/runner_output.py†L1-L154】

## Flip-rate (charge flip) measurement
- The flip runner enforces 5-tuples before materialising histogram summaries, matching the tuple-key standard.【F:analysis/flip_measurement/run_flip.py†L126-L151】

## MC validation
- The generator-level validation processor builds 5-element histogram tuples and labels aggregated outputs with optional application tags.【F:analysis/mc_validation/mc_validation_gen_processor.py†L299-L361】
- Plotting helpers normalise tuple keys to the 5-field standard and rely on stored application/channel/systematic values.【F:analysis/mc_validation/plot_utils.py†L19-L115】
