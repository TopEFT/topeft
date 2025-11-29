# Canonical jet layout fix for lj concatenation

## Summary
- `_ensure_object_collection_layout` now detects canonical jet struct-of-arrays inputs (presence of pt/eta/phi), zips their per-jet fields with `ak.zip(..., depth_limit=2)` to build `[events][jets]` lists, and keeps the existing protections for None inputs, singleton axes, and wrapper records. A debug-only log on `goodJets` captures the raw Awkward type when `_debug_logging` is enabled.
- Added `test_ensure_object_collection_layout_accepts_canonical_jets_struct_of_arrays` to exercise the struct-of-arrays conversion and confirmed the ambiguous wrapper test still raises. Existing lj concat regression keeps covering wrappers.

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" "$PYTHON_ENV" -m compileall analysis/topeft_run2` ✅
- `PYTHONNOUSERSITE=1 PYTHONPATH="" "$PYTHON_ENV" -m pytest -q tests/test_analysis_processor_variations.py -k "ensure_object_collection_layout or lj"` ✅
- `PYTHONNOUSERSITE=1 PYTHONPATH="" "$PYTHON_ENV" analysis/topeft_run2/run_analysis.py input_samples/cfgs/mc_signal_samples_NDSkim.cfg,input_samples/cfgs/mc_background_samples_NDSkim.cfg,input_samples/cfgs/data_samples_NDSkim.cfg --outname UL17_SRs_quickstart --outpath histos/local_debug --nworkers 1 --summary-verbosity brief --executor iterative --skip-cr --do-systs --log-level INFO -c 1 -s 5` → processed the previously failing UL17_tHq chunk (tasks ~228–229) without any layout errors; stopped after ~2 minutes to keep the iteration short.
