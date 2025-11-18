# Tuple key audit

This audit records where histogram tuple keys are created and consumed across the repository. The target convention is `(variable, channel, application, sample, systematic)`; any workflows still using four-field tuples or non-standard ordering are flagged for follow-up.

## Run 2 workflow
- `analysis/topeft_run2/analysis_processor.py` validates that `hist_keys` entries are 5-element tuples and builds keys as `(variable, channel, application, sample, systematic)`.【F:analysis/topeft_run2/analysis_processor.py†L322-L552】
- Downstream `HistogramCombination` usage in the Run 2 workflow therefore aligns with the 5-tuple standard.

## Training tutorial workflow
- The training processor builds 4-tuples `(variable, channel, sample, systematic)`, intentionally dropping the application component.【F:analysis/training/simple_processor.py†L322-L380】
- Tests enforce the 4-tuple shape and require no categorical axes in the stored histograms.【F:tests/test_training_tuple_output.py†L150-L171】
- The training runner expects 5-tuples and will raise if a histogram key is not `(variable, channel, application, sample, systematic)`, so it currently disagrees with the processor/tested output.【F:analysis/training/simple_run.py†L1-L104】

## Shared runner output helpers
- `topeft/modules/runner_output.py` assumes histogram payloads are keyed by 4-tuples `(variable, channel, sample, systematic)` when materialising summaries.【F:topeft/modules/runner_output.py†L40-L120】

## b-tag MC efficiency workflow
- Histograms are produced with keys `(variable, jet_flavour, working_point, sample, systematic)`, i.e. five components but without a channel/application split.【F:analysis/btagMCeff/btagMCeff.py†L140-L160】
- The runner script normalises outputs with `tuple_dict_stats`, which checks for 4-tuples, so this workflow mixes a non-standard 5-tuple producer with 4-tuple consumers.【F:analysis/btagMCeff/run.py†L125-L148】【F:topeft/modules/runner_output.py†L98-L120】

## Flip-rate (charge flip) measurement
- Both application-region and measurement-region processors emit 4-tuples `(variable, region, sample, systematic)` and summarise only 4-tuple entries.【F:analysis/flip_measurement/flip_ar_processor.py†L390-L436】【F:analysis/flip_measurement/run_flip.py†L132-L149】
- Plotting utilities iterate over tuple entries of length four and rely on string components instead of categorical axes.【F:analysis/flip_measurement/plot_utils.py†L1-L68】

## MC validation
- The generator-level validation processor collects 4-tuple histogram entries `(variable, channel, sample, systematic)` and aggregates them for plotting.【F:analysis/mc_validation/mc_validation_gen_processor.py†L306-L338】
- Plotting helpers rebuild categorical dataset/channel/systematic axes from the 4-tuple entries, embedding a categorical-axis reconstruction step rather than consuming stored categorical axes.【F:analysis/mc_validation/plot_utils.py†L36-L193】

## Summary of remaining gaps
- Training tutorial and shared runner helpers still assume 4-tuples, while the training runner advertises/validates 5-tuples.
- b-tag MC efficiency combines a 5-field (but non-standard) key with 4-tuple summary utilities.
- Flip-rate and MC validation workflows operate purely on 4-tuples without an application field; plotting tools encode categorical axes during reconstruction rather than in the stored histograms.
