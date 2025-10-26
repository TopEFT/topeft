# Remote environment maintenance

Changes to `environment.yml` affect both local development setups and the
archived environment shipped to remote workers. After editing the specification:

1. Recreate the local Conda environment so the lock file matches the new pins::

       conda env update -f environment.yml --prune

   This command updates the existing `coffea20250703` environment in-place and
   removes packages that are no longer required.

2. Regenerate the packaged tarball that workflows submit to remote resources::

       python -m topcoffea.modules.remote_environment

   The helper will compare the current `environment.yml` (including the
   `coffea==2025.7.3` / `awkward==2.8.7` pins) against the cached archive in
   `topeft-envs/` and build a fresh tarball when the spec or local editable
   packages (`topeft`, `topcoffea`) have changed.  It rewrites the pip section
   to `-e` installs automatically so the archive bundles the local checkouts
   referenced by `PIP_LOCAL_TO_WATCH`.

Remember to commit both the specification and the newly generated archive when
updating the shipped environment.
