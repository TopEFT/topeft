# MultiCell integration installation

The YAML supplies dependencies, not the unpublished integration commits. Obtain
both integrated checkouts before installing:

- TopEFT must contain `a16aa560f`, `75c149da7`, `5d93e1f05`, `dd4441701`.
- TopCoffea must contain `a669ac0`, `ab95b39`.

For each commit, check `git -C <path-to-repository> merge-base --is-ancestor
<commit> HEAD`. Record `git rev-parse HEAD` and `git status --short` in both
checkouts; a branch name alone does not establish the required content.

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1
conda env create -f <path-to-topeft>/environment-multicell.yml
conda activate coffea-multicell-env
python -m pip install --no-deps --no-build-isolation -e <path-to-topcoffea>
python -m pip install --no-deps --no-build-isolation -e <path-to-topeft>
python -c 'import sys, topcoffea, topeft; print(sys.executable); print(*sys.path, sep="\n"); print(topcoffea.__file__); print(topeft.__file__)'
git -C <path-to-topcoffea> rev-parse HEAD
git -C <path-to-topeft> rev-parse HEAD
```

Check that imports point to the two intended checkouts and that third-party
packages come from this environment, with no user-site or other checkout paths.
You may override the reusable environment name with `conda env create --name
<name> ...` or `--prefix <prefix>` and activate that name or prefix instead.

Run from the TopEFT root:

```bash
python -m pytest -q tests/test_run_analysis_hist_outputs.py tests/test_run_analysis_cli_help.py tests/test_run_analysis_preflight.py
python -c 'import runpy, sys; sys.path.insert(0, "analysis/topeft_run2"); runpy.run_path("analysis/topeft_run2/run_analysis.py", run_name="__main__")' --help
```

The pinned stack retains Coffea 0.7/Awkward 1 and supplies modern Hist MultiCell
storage, embedded second moments, Weight-backed SparseHist, raw counts, and
serialization. Both repositories' setup files leave dependencies unspecified,
so install the environment first. Coffea brings its own plotting dependencies;
no extra notebook or benchmark stack is requested.

The runner currently imports both a sibling `analysis_processor` and the
checkout-only `analysis` namespace (not installed by `setup.py`). The invocation
above makes both available from the checkout root without inherited `PYTHONPATH`.
Use the same invocation with normal runner arguments for local futures runs.
Shell campaign wrappers still invoke the script directly; their clean-environment
import failures require a separate packaging/invocation fix.
Matplotlib is pinned because artifact tests import plotting readers and newer
Matplotlib releases cannot import with this NumPy pin.

`ndcctools` and `conda-pack` retain the project's remote-executor tooling.
Validation targets local files with the futures executor; Work Queue/TaskVine
execution and remote environment shipping require separate validation. XRootD
is not a direct YAML requirement, but the solved tooling dependencies include
XRootD 6.1.1. Neither its import nor remote transport is validated; no claim of
segfault-free XRootD operation is made. A successful local ROOT-file smoke test
does not establish working XRootD transport.

The integration is producer-only: schema-v3 output has no physical `*_sumw2`
companions. Inline nonprompt postprocessing still requires historical scalar
companions, so `test_np_postprocess_inline_writes_transformed_artifact_sidecar`
remains a known failure in the full runner-output file. Do not bypass artifact
validation or synthesize legacy companions merely to make that test pass.
