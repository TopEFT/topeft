from collections.abc import Mapping
from coffea import processor


class TupleRunner(processor.Runner):
    """Runner variant that propagates tuple keys to processor metadata."""

    def run(self, fileset, processor_instance, treename=None, uproot_options=None, iteritems_options=None):
        # Normalize tuple keys before handing off to the parent Runner
        if isinstance(fileset, Mapping):
            tuple_map = {}
            normalized = {}
            for idx, (key, val) in enumerate(fileset.items()):
                if isinstance(key, tuple):
                    dataset_name = f"{key[3]}_{idx}"
                    normalized[dataset_name] = val
                    tuple_map[dataset_name] = key
                else:
                    normalized[key] = val
            self._tuple_map = tuple_map
            return super().run(
                normalized,
                processor_instance,
                treename=treename,
                uproot_options=uproot_options,
                iteritems_options=iteritems_options,
            )

        # No special handling needed
        self._tuple_map = {}
        return super().run(
            fileset,
            processor_instance,
            treename=treename,
            uproot_options=uproot_options,
            iteritems_options=iteritems_options,
        )

    def preprocess(self, fileset, treename=None):
        for chunk in super().preprocess(fileset, treename):
            ds = chunk.dataset
            tup = getattr(self, "_tuple_map", {}).get(ds, ds)
            if chunk.usermeta is None:
                chunk.usermeta = {}
            chunk.usermeta["tuple"] = tup
            yield chunk
