# Step 5W.1 – Wire run2_scenarios into the Run-2 workflow

## Summary
- Run-2 scenarios listed in `analysis/metadata/run2_scenarios.yaml` now source their channel groups through `run2_scenarios.load_run2_channels_for_scenario`, ensuring layouts match the canonical YAML.
- Non-Run-2 scenarios continue to rely on the inline `channels` block in their metadata files, so existing workflows behave as before.
- The workflow enforces a single primary scenario for channel planning and logs when additional names are supplied, preventing helper mismatches.
- Validator output confirms the expected SR-channel counts (TOP_22_006 = 43, tau_analysis = 68, offz_tau_analysis = 48, offz_fwd_analysis = 121, all_analysis = 148).

## Implementation details
- `analysis/topeft_run2/workflow.py` now determines the primary scenario before instantiating `ChannelMetadataHelper` and conditionally replaces `metadata['channels']` with the canonical loader output when applicable. A warning records any extra scenario names and canonical group counts are logged at INFO level.
- The new helper `is_run2_scenario`/`known_run2_scenarios` in `topeft/modules/run2_scenarios.py` exposes the scenario list cached from `run2_scenarios.yaml`, allowing the workflow to decide whether to invoke the loader without re-reading YAML.
- Failure to load canonical groups (e.g. because of a missing scenario definition) gracefully falls back to the inline metadata with a warning, keeping legacy scenarios untouched.

## Behavioural changes
- `scenario_registry.py` remains responsible for mapping scenario names to metadata bundles (variables, systematics, samples), while `run2_scenarios.py` is now the sole source of Run-2 channel group composition.
- `ChannelMetadataHelper` sees only the trimmed group set for canonical Run-2 scenarios, so histogram planning and executor summaries now match the validator counts.
- Non-Run-2 scenarios still pass their entire metadata-defined group list to the helper, preserving historic multi-scenario support where available.

## Key diff snippets
```diff
@@ analysis/topeft_run2/workflow.py @@
-    channels_metadata = metadata.get("channels")
-    if not channels_metadata:
-        raise ValueError(
-            f"Channel metadata is missing from the metadata YAML ({metadata_file})."
-        )
-
-    channel_helper = ChannelMetadataHelper(channels_metadata)
-    scenario_names = unique_preserving_order(config.scenario_names)
-    if not scenario_names:
-        scenario_names = [DEFAULT_SCENARIO_NAME]
-    config.scenario_names = list(scenario_names)
+    scenario_names = unique_preserving_order(config.scenario_names)
+    if not scenario_names:
+        scenario_names = [DEFAULT_SCENARIO_NAME]
+    config.scenario_names = list(scenario_names)
+    primary_scenario = config.scenario_names[0]
+
+    from topeft.modules import run2_scenarios
+
+    channels_metadata = metadata.get("channels")
+    channels_data = channels_metadata
+    use_run2_channels = run2_scenarios.is_run2_scenario(primary_scenario)
+    if use_run2_channels:
+        if len(config.scenario_names) > 1:
+            logger.warning(
+                "Run 2 scenario '%s' requested alongside additional scenarios (%s). Only the primary scenario can be used for channel selection; ignoring the rest.",
+                primary_scenario,
+                ", ".join(config.scenario_names[1:]),
+            )
+            config.scenario_names = [primary_scenario]
+        try:
+            channels_data = run2_scenarios.load_run2_channels_for_scenario(primary_scenario)
+            logger.info(
+                "Loaded %d Run 2 channel groups for scenario '%s'.",
+                len((channels_data or {}).get("groups", {})),
+                primary_scenario,
+            )
+        except Exception as exc:
+            logger.warning(
+                "Falling back to inline channel metadata for scenario '%s': %s",
+                primary_scenario,
+                exc,
+            )
+            channels_data = channels_metadata
+
+    if not channels_data:
+        raise ValueError(
+            f"Channel metadata is missing for scenario '{primary_scenario}'. Checked canonical Run 2 definitions and metadata YAML ({metadata_file})."
+        )
+
+    channel_helper = ChannelMetadataHelper(channels_data)
```

```diff
@@ topeft/modules/run2_scenarios.py @@
+def known_run2_scenarios() -> Tuple[str, ...]:
+    """Return the scenario names enumerated in ``run2_scenarios.yaml``."""
+
+    return tuple(_load_run2_scenarios().keys())
+
+
+def is_run2_scenario(name: str) -> bool:
+    """Return ``True`` when ``name`` is defined in ``run2_scenarios.yaml``."""
+
+    if not name:
+        return False
+    return name in _load_run2_scenarios()
```

## Testing
- `PYTHONNOUSERSITE=1 PYTHONPATH="" $PYTHON_ENV -m compileall analysis/topeft_run2/run_analysis.py analysis/topeft_run2/workflow.py analysis/topeft_run2/scenario_registry.py topeft/modules/run2_scenarios.py`
- `PYTHONNOUSERSITE=1 PYTHONPATH="" $PYTHON_ENV scratch/validate_run2_scenarios_step5.py`
  - Output:

    ```
    SCENARIO              GROUPS    SR_CHANNELS
    -------------------------------------------
    TOP_22_006                 2             43
    tau_analysis               3             68
    offz_tau_analysis          3             48
    offz_fwd_analysis          2            121
    all_analysis               3            148
    ```
