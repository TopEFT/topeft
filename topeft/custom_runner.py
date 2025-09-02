from collections.abc import Mapping
from coffea import processor


class TupleRunner(processor.Runner):
    """Runner variant that propagates tuple keys to processor metadata."""

    def run(self, fileset, processor_instance, treename=None, uproot_options=None, iteritems_options=None):
        if uproot_options is None:
            uproot_options = {}
        if iteritems_options is None:
            iteritems_options = {}

        # Allow fileset keys to be tuples and pass them through in metadata
        if isinstance(fileset, Mapping):
            tuple_map = {}
            normalized = {}
            for key, val in fileset.items():
                if isinstance(key, tuple):
                    dataset_name = key[0]
                    normalized[dataset_name] = val
                    tuple_map[dataset_name] = key
                else:
                    normalized[key] = val
            chunks = list(super().preprocess(normalized, treename))
            for chunk in chunks:
                ds = chunk.dataset
                tup = tuple_map.get(ds, ds)
                if chunk.usermeta is None:
                    chunk.usermeta = {}
                chunk.usermeta["tuple"] = tup
            return super().run(chunks, processor_instance, treename, uproot_options, iteritems_options)
        # fall back to default behaviour
        return super().run(fileset, processor_instance, treename, uproot_options, iteritems_options)
