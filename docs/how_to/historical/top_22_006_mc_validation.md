# Historical TOP-22-006 MC validation

> **Archival TOP-22-006 material.** The scripts remain under
> `analysis/mc_validation` and have not been updated since the August 2023
> `topcoffea` refactoring. They do not validate current TOP-26-006 samples.

The archived scripts compare Full Run 2 (FullR2) private Ultra Legacy (UL)
Monte Carlo (MC) samples generated for TOP-22-006 with the corresponding
central UL MC samples. These comparisons supported the June 2022 TOP-22-006
pre-approval checks.

`mc_validation_gen_processor.py` produces generator-level histograms for the
private-versus-central MC comparison.

`mc_validation_gen_plotter.py` reads those histograms and compares the private
and central generator-level distributions.

`mc_validation_plotter.py` reads `topeft` processor output and compares private
and central reconstructed-level distributions. It records the
reconstructed-level validation used for the June 2022 TOP-22-006 pre-approval
checks.

The generator-level comparison isolates differences already present in event
generation, while the reconstructed-level comparison includes reconstruction
and analysis-selection effects. The two views therefore answer different
validation questions and should not be treated as interchangeable.

This page records the original roles only. The exact June 2022 input datasets,
environment, and acceptance results are not reconstructed here. See the
[historical reproduction boundary](top_22_006.md) before consulting these
scripts.
