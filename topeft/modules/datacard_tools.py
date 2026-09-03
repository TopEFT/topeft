import pickle
import gzip
import numpy as np
import boost_histogram as bh
import uproot
import hist
import os
import re
import json
import time
import copy
import warnings

from collections import defaultdict

from topcoffea.modules.utils import regex_match, get_hist_from_pkl
from topeft.modules.paths import topeft_path
from topeft.modules.axes import info as axes_info
from topeft.modules.axes import info_2d as axes_info_2d
from topeft.modules.axis_binning import (
    BINNING_MODES,
    resolve_and_rebin_histogram,
    validate_matching_histogram_edges,
)
from topeft.modules.compatibility import add_sumw2_stub
from topeft.modules.data_driven_products import CANONICAL_DATA_DRIVEN_YEARS
from topeft.modules.histogram_artifact import (
    merge_histogram_sidecars,
    validate_histogram_artifact,
)
from topeft.modules.nominal_schema import (
    NOMINAL_CONTAINER_SCHEMA_VERSION,
    canonical_process_year,
    histogram_contribution_support,
    is_split_nominal_mapping,
    materialize_legacy_histogram_dict,
    merge_nominal_mappings,
    validate_year_coverage,
    validate_nominal_mapping,
    year_independent_process,
)
from topeft.modules.sumw2_policy import resolved_policy_from_provenance
from topeft.modules.missing_parton_contract import (
    DEFAULT_SR_REGISTRY,
    SR_CHANNEL_CONFIG_KEY,
    load_or_validate_selected_registry,
    load_missing_parton_channel_contract,
    validate_legacy_missing_parton_payload,
)


PRECISION = 6   # Decimal point precision in the text datacard output
SUMW2_SUFFIX = "_sumw2"
FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES = (
    "CMS_eff_t_DeepTau",
    "CMS_fake_t_DeepTau",
    "CMS_scale_t_DeepTau",
    "lepSF_taus_fake_run",
)


def resolve_shape_nuisance_identity(systematic, run_suffix, run_decorrelate):
    """Return one final nuisance base and paired template name for a variation."""
    direction = ""
    for candidate in ("Up", "Down"):
        if systematic.endswith(candidate):
            direction = candidate
            break
    base = systematic[: -len(direction)] if direction else systematic
    if base.startswith(FULLY_SPECIFIED_TAU_NUISANCE_PREFIXES):
        return base, f"{base}{direction}"
    if base in run_decorrelate:
        base = f"{base}{run_suffix}"
    return base, f"{base}{direction}"

# These process rules control which DatacardMaker templates retain stored bin
# variances in the ROOT output. All other templates are written with zero stored
# statistical variance.
stat_uncertainty_process_policy = {
    "exact_process_names": frozenset({"fakes"}),
    "process_name_substrings": ("close",),
}


def process_retains_stat_uncertainty(process_name):
    return (
        process_name in stat_uncertainty_process_policy["exact_process_names"]
        or any(
            substring in process_name
            for substring in stat_uncertainty_process_policy[
                "process_name_substrings"
            ]
        )
    )


def _is_sumw2_key(key):
    return isinstance(key, str) and key.endswith(SUMW2_SUFFIX)


def _base_key(key):
    return key[: -len(SUMW2_SUFFIX)]


def _short_examples(values, max_items=8):
    items = list(values)
    if len(items) <= max_items:
        return items
    return items[:max_items] + ["..."]


def _categorical_axis_names(h):
    cat_axes = getattr(h, "categorical_axes", None)
    if cat_axes is not None:
        try:
            return tuple(ax.name for ax in cat_axes)
        except Exception:
            cat_names = getattr(cat_axes, "name", None)
            if cat_names is None:
                raise
            if isinstance(cat_names, str):
                return (cat_names,)
            return tuple(cat_names)

    str_category_type = getattr(hist.axis, "StrCategory", None)
    int_category_type = getattr(hist.axis, "IntCategory", None)
    categorical_types = tuple(
        ax_type
        for ax_type in (str_category_type, int_category_type)
        if ax_type is not None
    )

    names = []
    for ax in h.axes:
        if categorical_types and isinstance(ax, categorical_types):
            names.append(ax.name)
        elif type(ax).__name__ in {"StrCategory", "IntCategory"}:
            names.append(ax.name)
    return tuple(names)


def _dense_axes(h):
    cat_names = set(_categorical_axis_names(h))
    return [ax for ax in h.axes if ax.name not in cat_names]


def _axis_edges(ax):
    if not hasattr(ax, "edges"):
        return None

    edges_obj = getattr(ax, "edges")
    try:
        edges = edges_obj() if callable(edges_obj) else edges_obj
    except Exception:
        return None

    try:
        arr = np.asarray(edges, dtype=float)
    except Exception:
        return None

    if arr.ndim != 1:
        return None
    return arr


def _validate_hist_compatibility(key, existing_hist, incoming_hist, incoming_path):
    if type(existing_hist) is not type(incoming_hist):
        raise TypeError(
            f"Histogram type mismatch for key '{key}' while merging '{incoming_path}': "
            f"{type(existing_hist)} != {type(incoming_hist)}"
        )

    if not hasattr(existing_hist, "axes") or not hasattr(incoming_hist, "axes"):
        raise TypeError(
            f"Object for key '{key}' from '{incoming_path}' does not look like a histogram "
            f"(missing 'axes')."
        )

    existing_all_axis_names = tuple(ax.name for ax in existing_hist.axes)
    incoming_all_axis_names = tuple(ax.name for ax in incoming_hist.axes)
    if existing_all_axis_names != incoming_all_axis_names:
        raise ValueError(
            f"Axis-name/order mismatch for key '{key}' while merging '{incoming_path}': "
            f"{existing_all_axis_names} != {incoming_all_axis_names}"
        )

    existing_cat_axes_obj = getattr(existing_hist, "categorical_axes", [])
    incoming_cat_axes_obj = getattr(incoming_hist, "categorical_axes", [])
    existing_cat_axes = tuple((ax.name, type(ax).__name__) for ax in existing_cat_axes_obj)
    incoming_cat_axes = tuple((ax.name, type(ax).__name__) for ax in incoming_cat_axes_obj)
    if existing_cat_axes != incoming_cat_axes:
        raise ValueError(
            f"Categorical axis mismatch for key '{key}' while merging '{incoming_path}': "
            f"{existing_cat_axes} != {incoming_cat_axes}"
        )

    existing_dense_axes = _dense_axes(existing_hist)
    incoming_dense_axes = _dense_axes(incoming_hist)
    if len(existing_dense_axes) != len(incoming_dense_axes):
        raise ValueError(
            f"Dense-axis-count mismatch for key '{key}' while merging '{incoming_path}': "
            f"{len(existing_dense_axes)} != {len(incoming_dense_axes)}"
        )

    for existing_axis, incoming_axis in zip(existing_dense_axes, incoming_dense_axes):
        if type(existing_axis) is not type(incoming_axis):
            raise ValueError(
                f"Dense-axis type mismatch for key '{key}' axis '{existing_axis.name}' while merging "
                f"'{incoming_path}': {type(existing_axis)} != {type(incoming_axis)}"
            )
        if existing_axis.name != incoming_axis.name:
            raise ValueError(
                f"Dense-axis name mismatch for key '{key}' while merging '{incoming_path}': "
                f"{existing_axis.name} != {incoming_axis.name}"
            )
        if len(existing_axis) != len(incoming_axis):
            raise ValueError(
                f"Dense-axis bin-count mismatch for key '{key}' axis '{existing_axis.name}' while "
                f"merging '{incoming_path}': {len(existing_axis)} != {len(incoming_axis)}"
            )
        existing_edges = _axis_edges(existing_axis)
        incoming_edges = _axis_edges(incoming_axis)
        if existing_edges is not None and incoming_edges is not None:
            if existing_edges.shape != incoming_edges.shape or not np.array_equal(existing_edges, incoming_edges):
                raise ValueError(
                    f"Dense-axis edges mismatch for key '{key}' axis '{existing_axis.name}' while "
                    f"merging '{incoming_path}'."
                )

    existing_wcs = getattr(existing_hist, "wc_names", None)
    incoming_wcs = getattr(incoming_hist, "wc_names", None)
    if existing_wcs is not None or incoming_wcs is not None:
        if list(existing_wcs) != list(incoming_wcs):
            raise ValueError(
                f"WC-name mismatch for key '{key}' while merging '{incoming_path}': "
                f"{existing_wcs} != {incoming_wcs}"
            )


def _process_labels(hist_obj):
    try:
        return set(hist_obj.axes["process"])
    except Exception:
        return None


_RUN2_CANONICAL_YEARS = frozenset(
    year for year in CANONICAL_DATA_DRIVEN_YEARS if year.startswith("UL")
)
_RUN3_CANONICAL_YEARS = frozenset(CANONICAL_DATA_DRIVEN_YEARS) - _RUN2_CANONICAL_YEARS


def _canonical_run_eras(processes):
    eras = set()
    for process in processes:
        year = canonical_process_year(str(process))
        if year in _RUN2_CANONICAL_YEARS:
            eras.add("run2")
        elif year in _RUN3_CANONICAL_YEARS:
            eras.add("run3")
    return frozenset(eras)


def _mapping_process_labels(histograms):
    return {
        process
        for histogram in histograms.values()
        for process in (_process_labels(histogram) or ())
    }


def _reject_cross_run_histogram_composition(
    pkl_paths,
    loaded_inputs,
    input_metadata,
):
    input_eras = []
    for path, histograms, metadata in zip(
        pkl_paths, loaded_inputs, input_metadata
    ):
        if metadata is None:
            processes = _mapping_process_labels(histograms)
        else:
            processes = metadata["sumw2_storage_provenance"][
                "resolved_processes"
            ]
        input_eras.append((path, _canonical_run_eras(processes)))

    composed_eras = set().union(*(eras for _path, eras in input_eras))
    if {"run2", "run3"} <= composed_eras:
        era_summary = ", ".join(
            f"{path}={sorted(eras)}" for path, eras in input_eras
        )
        raise RuntimeError(
            "Histogram-level Run 2 + Run 3 composition is unsupported. "
            "Produce Run 2 and Run 3 cards separately and combine them only at "
            "the card/workspace/statistical-model level. "
            f"Resolved input eras: {era_summary}."
        )


def _validate_disjoint_histogram_contributions(pkl_paths, loaded_inputs):
    owner_by_contribution = {}
    for path, histograms in zip(pkl_paths, loaded_inputs):
        contributions = histogram_contribution_support(histograms)
        collisions = sorted(
            set(owner_by_contribution) & contributions,
            key=repr,
        )
        if collisions:
            examples = _short_examples(collisions, max_items=12)
            first_owner = owner_by_contribution[collisions[0]]
            raise RuntimeError(
                "Duplicate histogram contribution support detected before "
                f"histogram addition: incoming_path={path!r}, "
                f"existing_path={first_owner!r}, contributions={examples!r}. "
                "A contribution is identified by its payload component key and "
                "complete categorical coordinate."
            )
        owner_by_contribution.update(
            {contribution: path for contribution in contributions}
        )


def load_and_merge_histogram_pkls(
    pkl_paths,
    *,
    require_sumw2=True,
    consumer_required_families=(),
    year_coverage_policy="off",
):
    """
    Load and validate one or more histogram PKLs before merging them in memory.

    Every input owns exact structural contributions identified by payload
    component key plus the complete categorical coordinate.  Channel and
    process labels may repeat when another coordinate differs.

    Returns: (merged_hist_dict, merge_report_dict)
    """
    if not pkl_paths:
        raise ValueError("No input pickle files were provided for merging.")
    validate_year_coverage({}, policy=year_coverage_policy)

    report = {
        "num_inputs": len(pkl_paths),
        "inputs": list(pkl_paths),
        "require_sumw2": bool(require_sumw2),
        "year_coverage_policy": year_coverage_policy,
        "files": [],
        "contribution_identity": "payload_component_key + complete_categorical_coordinate",
        "schema": None,
    }

    loaded_inputs = []
    input_metadata = []
    legacy_inputs = []
    for path in pkl_paths:
        print(f"Opening: {path}")
        tic = time.time()
        hist_dict = get_hist_from_pkl(path, allow_empty=False)
        dt = time.time() - tic
        print(f"Pkl Open Time: {dt:.2f} s")

        if not isinstance(hist_dict, dict):
            raise TypeError(
                f"Histogram input '{path}' is not a dictionary. Got: {type(hist_dict)}"
            )
        non_string_keys = [k for k in hist_dict if not isinstance(k, str)]
        if non_string_keys:
            raise TypeError(
                f"Histogram input '{path}' contains non-string keys: "
                f"{_short_examples(non_string_keys)}"
            )
        artifact_validation = validate_histogram_artifact(path, hist_dict)
        metadata = artifact_validation["metadata"]
        if artifact_validation["schema"] == "legacy_uniform":
            legacy_inputs.append(path)
        loaded_inputs.append(hist_dict)
        input_metadata.append(metadata)

    if legacy_inputs:
        warnings.warn(
            "Loading legacy uniform histogram PKL(s) without schema-v2 sidecars through "
            "the explicit compatibility path: " + ", ".join(legacy_inputs),
            UserWarning,
            stacklevel=2,
        )

    schema_versions = {
        None
        if metadata is None
        else metadata["artifact"]["nominal_container_schema_version"]
        for metadata in input_metadata
    }
    if len(schema_versions) != 1:
        raise RuntimeError("Cannot merge legacy and versioned nominal schemas together.")
    schema_version = next(iter(schema_versions))
    report["schema"] = "legacy_uniform" if schema_version is None else "split_sibling_v1"

    _reject_cross_run_histogram_composition(
        pkl_paths,
        loaded_inputs,
        input_metadata,
    )

    merged_hists = {}
    if schema_version == NOMINAL_CONTAINER_SCHEMA_VERSION:
        merged_sidecar = merge_histogram_sidecars(input_metadata)
        if len(pkl_paths) > 1:
            _validate_disjoint_histogram_contributions(pkl_paths, loaded_inputs)
        artifact_kind = merged_sidecar["artifact_kind"]
        policy = resolved_policy_from_provenance(
            merged_sidecar["sumw2_storage_provenance"]
        )
        runtime_families = policy.runtime_histogram_families
        required_families = frozenset(consumer_required_families)
        if artifact_kind == "processor_output":
            missing_policy_requirements = sorted(
                family
                for family in required_families
                if not policy.selects_family(family)
            )
        else:
            missing_policy_requirements = sorted(
                family
                for family in required_families
                if not merged_sidecar["required_sumw2_processes"].get(family)
            )
        if missing_policy_requirements:
            raise RuntimeError(
                "Active consumer requirements are absent from the artifact contract: "
                + ", ".join(missing_policy_requirements)
            )
        for path, hist_dict, metadata in zip(
            pkl_paths, loaded_inputs, input_metadata
        ):
            input_policy = resolved_policy_from_provenance(
                metadata["sumw2_storage_provenance"]
            )
            input_runtime_families = input_policy.runtime_histogram_families
            validate_nominal_mapping(
                hist_dict,
                runtime_families=input_runtime_families,
                schema_version=schema_version,
                policy=(
                    input_policy if artifact_kind == "processor_output" else None
                ),
            )
            keys = set(hist_dict)
            report["files"].append(
                {
                    "path": path,
                    "num_keys": len(keys),
                    "num_base_keys": len(input_runtime_families),
                    "num_sumw2_keys": sum(_is_sumw2_key(key) for key in keys),
                }
            )
        merged_hists = merge_nominal_mappings(
            loaded_inputs,
            runtime_families=runtime_families,
            schema_version=schema_version,
            policy=policy if artifact_kind == "processor_output" else None,
        )
        report["sumw2_storage_provenance"] = policy.to_provenance()
        report["production_sample_contract"] = merged_sidecar[
            "production_sample_contract"
        ]
        report["runtime_histogram_families"] = list(runtime_families)
        report["artifact_kind"] = artifact_kind
        report["artifact_merged"] = True
        report["required_sumw2_processes"] = merged_sidecar[
            "required_sumw2_processes"
        ]
        report["transformation_contract"] = merged_sidecar[
            "transformation_contract"
        ]
        report["requested_data_driven_products"] = merged_sidecar[
            "requested_data_driven_products"
        ]
        report["resolved_data_driven_contract"] = merged_sidecar[
            "resolved_data_driven_contract"
        ]
        report["lineage_inputs"] = merged_sidecar["lineage_inputs"]
    elif schema_version is None:
        report["artifact_kind"] = "legacy_uniform"
        report["artifact_merged"] = len(pkl_paths) > 1
        if len(pkl_paths) > 1:
            _validate_disjoint_histogram_contributions(pkl_paths, loaded_inputs)
        for path, hist_dict in zip(pkl_paths, loaded_inputs):
            keys = set(hist_dict.keys())
            base_keys = sorted(k for k in keys if not _is_sumw2_key(k))
            sumw2_keys = sorted(k for k in keys if _is_sumw2_key(k))
            orphan_sumw2 = sorted(k for k in sumw2_keys if _base_key(k) not in keys)
            if orphan_sumw2:
                raise RuntimeError(
                    f"Input '{path}' contains *_sumw2 keys without base histograms: "
                    f"{_short_examples(orphan_sumw2)}"
                )
            if require_sumw2:
                missing_sumw2 = sorted(
                    key for key in base_keys if f"{key}{SUMW2_SUFFIX}" not in keys
                )
                if missing_sumw2:
                    raise RuntimeError(
                        f"Input '{path}' is missing required *_sumw2 companions for: "
                        f"{_short_examples(missing_sumw2)}"
                    )
            report["files"].append(
                {
                    "path": path,
                    "num_keys": len(keys),
                    "num_base_keys": len(base_keys),
                    "num_sumw2_keys": len(sumw2_keys),
                }
            )
            for key, incoming_hist in hist_dict.items():
                if key not in merged_hists:
                    merged_hists[key] = copy.deepcopy(incoming_hist)
                    continue
                existing_hist = merged_hists[key]
                _validate_hist_compatibility(key, existing_hist, incoming_hist, path)
                merged_hists[key] += incoming_hist
    else:
        raise RuntimeError(f"Unsupported nominal schema version {schema_version!r}.")

    report["num_merged_keys"] = len(merged_hists)
    report["year_coverage_mismatches"] = validate_year_coverage(
        merged_hists,
        policy=year_coverage_policy,
    )
    report["num_year_coverage_mismatches"] = len(
        report["year_coverage_mismatches"]
    )

    return merged_hists, report

def to_hist(arr,name,zero_wgts=False):
    """
        Converts a numpy array into a hist.Hist object suitable for being written to a root file by
        uproot. If 'zero_wgts' is true, then the resulting histogram will be created with bin errors
        set to 0 (instead of left unset)
    """
    # NOTE:
    #   If we don't instantiate a new np.array here, then clipped will store a reference to the
    #   sub-array arr and when we modify clipped, it will propagate back to arr as well!
    clipped = []
    for i in range(2):  # first entry is sum(weight), second entry is sum(weight^2)
        if arr[i] is not None:
            clipped.append(np.array(arr[i][1:]))  # Strip off the underoverflow bin
        else:
            clipped[i] = None

    nbins = len(clipped[0])
    h = hist.Hist(hist.axis.Regular(nbins,0,nbins,name=name),storage=bh.storage.Weight())
    if zero_wgts:
        h[...] = np.stack([clipped[0],np.zeros_like(clipped[0])],axis=-1) # Set the bin errors all to 0
    else:
        h[...] = np.stack([clipped[0], clipped[1]],axis=-1)
    return h


def _sanitize_negative_template_bins(templates):
    """Crop negative bins without letting variations revive raw-negative nominal bins."""
    raw_nominal = next(
        (
            arr
            for sp_key, arr in templates.items()
            if sp_key.systematic == "nominal"
        ),
        None,
    )
    nominal_negative_mask = (
        None if raw_nominal is None else np.asarray(raw_nominal[0]) < 0
    )

    sanitized_templates = {}
    for sp_key, arr in templates.items():
        sanitized_arr = [
            None if component is None else np.array(component, copy=True)
            for component in arr
        ]
        if nominal_negative_mask is not None:
            for component in sanitized_arr:
                if component is not None:
                    component[nominal_negative_mask] = 0

        negative_bin_mask = sanitized_arr[0] < 0
        sanitized_arr[0][negative_bin_mask] = 0
        if sanitized_arr[1] is not None:
            sanitized_arr[1][negative_bin_mask] = 0
        sanitized_templates[sp_key] = sanitized_arr

    return sanitized_templates


def _validate_ff_template_support(
    templates,
    *,
    variable,
    channel,
    process,
    decomposition,
):
    """Require FF shape support to be compatible with the sanitized nominal."""
    ff_templates = [
        (sp_key, arr)
        for sp_key, arr in templates.items()
        if "FF" in sp_key.systematic
    ]
    if not ff_templates:
        return

    nominal_templates = [
        (sp_key, arr)
        for sp_key, arr in templates.items()
        if sp_key.systematic == "nominal"
    ]
    if len(nominal_templates) != 1:
        raise ValueError(
            "FF template support validation requires exactly one sanitized nominal "
            f"template, found {len(nominal_templates)} "
            f"(variable={variable!r}, channel={channel!r}, process={process!r}, "
            f"decomposition={decomposition!r})."
        )

    nominal_key, nominal_arr = nominal_templates[0]
    if np.sum(nominal_arr[0]) != 0:
        return

    for variation_key, variation_arr in ff_templates:
        if np.sum(variation_arr[0]) != 0:
            raise ValueError(
                "FF template support mismatch: sanitized nominal content is zero "
                "but the FF variation content is nonzero "
                f"(variable={variable!r}, channel={channel!r}, process={process!r}, "
                f"decomposition={decomposition!r}, nominal_key={nominal_key!r}, "
                f"variation_key={variation_key!r})."
            )


class RateSystematic():
    def __init__(self,name,**kwargs):
        self.all = kwargs.pop("all",False)      # If true, this syst applies to all processes
        if self.all:
            try:
                self.all_unc = kwargs.pop("unc")
            except KeyError:
                msg = "Missing 'unc' argument. Must specify an uncertainty when using the 'all' option"
                raise KeyError(msg)
        self.name = name
        self.corrs = {}  # keys are the name of processes and values are the corresponding unc.

    def has_process(self,p):
        return self.all or (p in self.corrs)

    def add_process(self,p,v=None):
        if self.all:
            raise KeyError("Can't add a correlated process for systematic defined with the 'all' option")
        self.corrs[p] = v

    # TODO: This needs to be given a better name
    # Returns the corresponding unc. (i.e. kappa values) that have been associated with a particular process
    # Note: The return value should be as a string
    def get_process(self,p):
        if self.all:
            return self.all_unc
        if self.has_process(p):
            return self.corrs[p]
        else:
            # This is the case for a systematic that doesn't apply to the specified process
            return '-'

class JetScale(RateSystematic):
    def __init__(self,name,**kwargs):
        super().__init__(name,**kwargs)

        self.symmeterize = True     # whether or not we attempt to make the up/down shifts equal in absolute terms
        self.min_lo = 0.01          # For large kappa values, do not let the symmeterization go negative

    # Override the base implementation to handle the different dict structure
    # Note: The return value should be as a string
    def get_process(self,p,j):
        j = str(j)
        if self.all:
            unc_hi = self.all_unc[j]
            if self.symmeterize:
                unc_lo = max(self.min_lo,2 - unc_hi)
                return f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"
            else:
                return f"{unc_hi:.{PRECISION}f}"
        if self.has_process(p):
            unc_hi = self.corrs[p][j]
            if self.symmeterize:
                unc_lo = max(self.min_lo,2 - unc_hi)
                return f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"
            else:
                return f"{unc_hi:.{PRECISION}f}"
        else:
            return '-'

class MissingParton(RateSystematic):
    # Maps channel name from pkl file to hist name in missing_parton.root file
    CH_MAP = {
        "2los_onZ": "2los_onZ_1tau",
        "2lss_4t_m": "2lss_4t_m_2b",
        "2lss_4t_p": "2lss_4t_p_2b",
        "2lss_m": "2lss_m_2b",
        "2lss_p": "2lss_p_2b",
        "2lss_fwd_m": "2lss_fwd_m_2b",
        "2lss_fwd_p": "2lss_fwd_p_2b",
        "3l_onZ_1b": "3l_sfz_1b",
        "3l_onZ_2b": "3l_sfz_2b",
        "3l_p_offZ_1b": "3l1b_p",
        "3l_m_offZ_1b": "3l1b_m",
        "3l_p_offZ_2b": "3l2b_p",
        "3l_m_offZ_2b": "3l2b_m",
        "4l_2b": "4l",
    }

    def __init__(self,name,**kwargs):
        super().__init__(name,**kwargs)

    # Override the base implementation to handle the different dict structure
    def get_process(self,p,ch,l,j,b):
        pass

class DatacardMaker():
    # TODO:
    #   We are abusing the grouping mechanism to also handle renaming processes, but might want to
    #   separate into two distinct actions to make things easier to follow for the reader
    # Note:
    #   Care must be taken with regards to the underscores, due to 'nonprompt', 'data', and 'flips'
    GROUP= {
        "Diboson_": [
            "WZTo3LNu_",
            "WZto3LNu-2Jets_", #ttll 4to10
            "WWTo2L2Nu_",
            "ZZTo4L_",
            "ggToZZTo2e2mu_",
            "ggToZZTo4e_",
            "ggToZZTo2e2tau_",
            "ggToZZTo4tau_",
            "ggToZZTo4mu_",
            "ggToZZTo2mu2tau_",
        ],
        "Triboson_": [
            "WWW_",
            "WWZ_",
            "WZZ_",
            "ZZZ_",
        ],
        "tWZ": [
            "TWZToLL_",
            "TWZ_Tto2Q_WtoLNu_",
            "TWZ_TtoLNu_WtoLNu_",
            "TWZ_TtoLNu_Wto2Q_",
            "TWZ_TtoLNu_Wto2Q_Zto2L_",
            "TWZ_Tto2Q_WtoLNu_Zto2L_",
            "TWZ_TtoLNu_WtoLNu_Zto2L_",


        ],
        "convs": [
            "TTGamma_",
            "TTGJets_",
            "TTG-1Jets_",
            "TTG-1Jets_PTG-100to200_",
            "TTG-1Jets_PTG-10to100_",
            "TTG-1Jets_PTG-200_",
        ],
        "fakes": ["nonprompt"],
        "charge_flips_": ["flips"],
        "data_obs": ["data"],

        "ttH_": [
            "ttHJet_",
            "ttH_",
        ],
        "ttll_": [
            "ttllJet_",
            "TTZToLL_M1to10_",
            "TTToSemiLeptonic_",
            "TTTo2L2Nu_",
        ],
        "ttlnu_": [
            "ttlnuJet_",
            "ttlnu_",
        ],
    }

    YEARS = ["UL16","UL16APV","UL17","UL18","2022","2022EE","2023","2023BPix"]

    SYST_YEARS = ["2016","2016APV","2017","2018","2022","2022EE","2023","2023BPix"]

    MISSING_PARTON_YEAR_ERAS = {
        "UL16": "run2",
        "UL16APV": "run2",
        "UL17": "run2",
        "UL18": "run2",
        "2022": "run3",
        "2022EE": "run3",
        "2023": "run3",
        "2023BPix": "run3",
    }
    MISSING_PARTON_NUISANCE_NAME = "missing_parton"
    MISSING_PARTON_DEFAULT_PAYLOADS = {
        "run2": "data/missing_parton/missing_parton_run2.root",
        "run3": "data/missing_parton/missing_parton_run3.root",
    }

    FNAME_TEMPLATE = "ttx_multileptons-{cat}_{kmvar}.{ext}"
    # FNAME_TEMPLATE = "TESTING_ttx_multileptons-{cat}.{ext}"

    SIGNALS = set(["ttH","tllq","ttll","ttlnu","tHq","tttt"])

    @classmethod
    def get_year(cls,s):
        """
            Attempt to return the year of the process or systematic string
        """
        for yr in cls.YEARS:
            if s.endswith(yr): return yr
        for yr in cls.SYST_YEARS:
            if s.endswith(yr+"Up"): return yr
            if s.endswith(yr+"Down"): return yr
        return None

    @classmethod
    def missing_parton_run_era(cls, year_or_period):
        """Resolve one canonical card-making period to its missing-parton era."""
        if not isinstance(year_or_period, str) or not year_or_period:
            raise ValueError(
                "Missing canonical year or period for missing-parton nuisance correlation."
            )
        try:
            return cls.MISSING_PARTON_YEAR_ERAS[year_or_period]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported canonical year or period {year_or_period!r} for "
                "missing-parton nuisance correlation. Supported values: "
                f"{tuple(cls.MISSING_PARTON_YEAR_ERAS)!r}."
            ) from exc

    @classmethod
    def missing_parton_nuisance_name(cls, year_or_period):
        cls.missing_parton_run_era(year_or_period)
        return cls.MISSING_PARTON_NUISANCE_NAME

    @classmethod
    def missing_parton_run_era_for_years(cls, year_or_periods, payload_path=None):
        if isinstance(year_or_periods, str):
            year_or_periods = (year_or_periods,)
        else:
            try:
                year_or_periods = tuple(year_or_periods)
            except TypeError as exc:
                raise ValueError(
                    "Missing canonical year or period for missing-parton nuisance correlation."
                ) from exc
        if not year_or_periods:
            raise ValueError(
                "Missing canonical year or period for missing-parton nuisance correlation."
            )

        resolved_eras = tuple(
            cls.missing_parton_run_era(year_or_period)
            for year_or_period in year_or_periods
        )
        unique_eras = set(resolved_eras)
        if len(unique_eras) != 1:
            payload_text = ""
            if payload_path is not None:
                payload_text = f" Active payload path: {payload_path!r}."
            raise ValueError(
                "Mixed Run 2 and Run 3 years cannot use one explicit missing-parton "
                "payload source in a single DatacardMaker run: "
                f"original labels={year_or_periods!r}, resolved eras={resolved_eras!r}. "
                "Produce Run 2 and Run 3 cards separately with their matching payloads."
                f"{payload_text}"
            )
        return resolved_eras[0]

    @classmethod
    def missing_parton_nuisance_name_for_years(cls, year_or_periods, payload_path=None):
        cls.missing_parton_run_era_for_years(
            year_or_periods,
            payload_path=payload_path,
        )
        return cls.MISSING_PARTON_NUISANCE_NAME

    @classmethod
    def resolve_missing_parton_payload_path(cls, year_or_periods, payload_path=None, sr_registry=DEFAULT_SR_REGISTRY):
        if payload_path == "":
            raise ValueError(
                "An explicit missing-parton payload path must be non-empty. Omit the "
                "option to select the run-era default."
            )
        registry, _ = load_or_validate_selected_registry(sr_registry)
        era = cls.missing_parton_run_era_for_years(
            year_or_periods,
            payload_path=payload_path,
        )
        if payload_path is not None:
            return payload_path
        if registry != DEFAULT_SR_REGISTRY:
            raise ValueError(
                f"Selected SR registry {registry!r} has no canonical implicit missing-parton payload. "
                "Use --miss-parton-file with a payload generated for the same registry."
            )
        return cls.MISSING_PARTON_DEFAULT_PAYLOADS[era]

    @classmethod
    def is_missing_parton_nuisance_name(cls, nuisance_name):
        return nuisance_name == cls.MISSING_PARTON_NUISANCE_NAME

    @classmethod
    def strip_fluctuation(cls,s):
        return s.replace("Down","").replace("Up","")

    @classmethod
    #def strip_year(cls,s):
    #    for yr in cls.YEARS:
    #        s = s.replace(yr,"")
    #    for yr in cls.SYST_YEARS:
    #        s = s.replace(f"_{yr}","")  # Note the underscore
    #    return s

    def strip_year(cls, s):
        # Remove longer year strings first
        for yr in sorted(cls.SYST_YEARS, key=len, reverse=True):
            s = s.replace(f"_{yr}", "")
        for yr in sorted(cls.YEARS, key=len, reverse=True):
            s = s.replace(yr, "")
        return s

    @classmethod
    def is_signal(cls,s):
        s = cls.get_process(s)
        return (s in cls.SIGNALS)

    @classmethod
    def is_per_year_systematic(cls,s):
        end_chks = [
            "_2016APVUp","_2016Up","_2017Up","_2018Up",
            "_2016APVDown","_2016Down","_2017Down","_2018Down","_2022Up", "_2022Down", "_2022EEUp", "_2022EEDown", "_2023Up", "_2023Down", "_2023BPixUp", "_2023BPixDown"
        ]
        return any([s.endswith(x) for x in end_chks])

    @classmethod
    def is_eft_term(cls,s):
        """ Check if string corresponds an EFT process term after decomposition."""
        chks = ["_lin_","_quad_"]
        return any([x in s for x in chks])

    @classmethod
    def get_process(cls,s):
        """ Strips off the year designation from a process name, can also be used for decomposed terms."""
        return year_independent_process(s)

    # TODO: I don't like the naming
    @classmethod
    def get_jet_mults(cls,s):
        """
            Returns the njet and bjet multiplicities based on the string passed to it in (j,b) order.
            For the regular expression, group 1 matches 'njet_bjet', group 2 matches 'bjet_njet'
            group 3 matches '_njet'.
        """
        rgx = re.compile(r"(_[1-7]j_[1-2]b)|(_[1-2]b_[1-7]j)|(_[1-7]j$)")

        m = rgx.search(s)
        if m.group(1) and m.group(2) is None and m.group(3) is None:
            # The order is '_Nj_Mb'
            _,j,b = m.group(1).split("_")
        elif m.group(1) is None and m.group(2) and m.group(3) is None:
            # The order is '_Nb_Mj'
            _,b,j = m.group(2).split("_")
        elif m.group(1) is None and m.group(2) is None and m.group(3):
            # This occurs when the string ends in '_Mj' and doesn't have a bjet multiplicity
            b = None
            j = m.group(3).replace("_","")
        else:
            raise ValueError(f"Unable to find rgx match in string {s}")
        j = int(j.replace("j",""))
        if b is not None:
            b = int(b.replace("b",""))
        return (j,b)

    @classmethod
    def get_lep_mult(cls,s):
        """ Returns the lepton multiplicity based on the string passed to it."""
        if s.startswith("2l"):
            return 2
        elif s.startswith("3l_"):
            return 3
        elif s.startswith("4l"):
            return 4
        else:
            raise ValueError(f"Unable to determine lepton multiplicity from string {s}")

    @staticmethod
    def _axis_names(h):
        if h is None:
            return ()
        try:
            return tuple(ax.name for ax in h.axes)
        except Exception:
            axes_name = getattr(getattr(h, "axes", None), "name", None)
            if axes_name is None:
                return ()
            if isinstance(axes_name, str):
                return (axes_name,)
            return tuple(axes_name)

    def _get_missing_parton_channel_contract(self):
        contract = getattr(self, "_missing_parton_channel_contract", None)
        if contract is None:
            contract = load_missing_parton_channel_contract()
            self._missing_parton_channel_contract = contract
        return contract

    def _resolve_supported_sr_appl(self, h, channel, process=None):
        if h is None or "appl" not in self._axis_names(h):
            return None

        contract = self._get_missing_parton_channel_contract()
        process_text = "" if process is None else f", process {process!r}"
        try:
            expected_sr_appl = contract.expected_sr_appl(channel)
        except ValueError as exc:
            raise ValueError(
                "DatacardMaker application-axis selection currently supports only "
                "metadata-defined SR channels. Requested channel "
                f"{channel!r}{process_text} is not in the "
                f"{SR_CHANNEL_CONFIG_KEY} contract. CR/AR application-axis card "
                "production is not implemented. No SR/AR integration, label "
                "guessing, or fallback was performed. Use an already projected/"
                "no-appl input or add a separately reviewed supported workflow."
            ) from exc

        available_appl = [str(label) for label in h.axes["appl"]]
        if expected_sr_appl not in available_appl:
            raise ValueError(
                f"DatacardMaker recognized channel {channel!r}{process_text} as a "
                "metadata-defined SR channel, but its exact expected appl label "
                f"{expected_sr_appl!r} is missing. Available appl labels are "
                f"{available_appl!r}. No SR/AR integration, label guessing, or "
                "fallback was performed."
            )
        return expected_sr_appl

    def select_final_sr_appl(self, h, channel, process=None):
        """Select the metadata-defined SR appl category when the axis exists."""
        expected_sr_appl = self._resolve_supported_sr_appl(
            h,
            channel,
            process=process,
        )
        if expected_sr_appl is None:
            return h

        return h.integrate("appl", expected_sr_appl)

    @staticmethod
    def _sparse_key_mapping(sp_key):
        if hasattr(sp_key, "_asdict"):
            return dict(sp_key._asdict())
        fields = getattr(sp_key, "_fields", None)
        if fields is not None:
            return dict(zip(fields, sp_key))
        return {}

    @classmethod
    def validate_sparse_axes_for_card(cls, templates, channel, process):
        """Reject unresolved sparse axes that would duplicate template names."""
        labels_by_axis = defaultdict(set)
        for sp_key in templates:
            key_map = cls._sparse_key_mapping(sp_key)
            for axis_name, label in key_map.items():
                if axis_name != "systematic":
                    labels_by_axis[axis_name].add(str(label))

        duplicate_producing_axes = {
            axis_name: labels
            for axis_name, labels in labels_by_axis.items()
            if len(labels) > 1
        }
        if not duplicate_producing_axes:
            return

        axis_name = sorted(duplicate_producing_axes)[0]
        labels = sorted(duplicate_producing_axes[axis_name])
        raise ValueError(
            f"Unresolved sparse axis {axis_name!r} while writing datacard channel "
            f"{channel!r}, process {process!r}; labels are {labels!r}. This would "
            "create duplicate ROOT template names. Resolve the sparse category "
            "before card writing."
        )

    @classmethod
    def get_processes_by_years(cls,h):
        """
            Reads the 'process' sparse axis of a histogram and returns a dictionary that maps stripped
            process names to the list of sparse axis categories it came from.
        """
        r = defaultdict(lambda: [])
        for x in h.axes["process"]:
            p = cls.get_process(x)
            r[p].append(cls.get_process(x))
        return r

    def __init__(self,pkl_path=None,hists=None,**kwargs):
        self.binning_mode    = kwargs.pop("binning_mode", "fitting")
        if self.binning_mode not in BINNING_MODES:
            raise ValueError(
                f"Unknown binning mode {self.binning_mode!r}; expected one of {BINNING_MODES}."
            )
        self.year_lst        = kwargs.pop("year_lst",[])
        self.do_sm           = kwargs.pop("do_sm",False)
        self.do_nuisance     = kwargs.pop("do_nuisance",False)
        self.drop_syst       = kwargs.pop("drop_syst",[])
        self.skip_missing_parton_rate_syst = bool(
            kwargs.pop("skip_missing_parton_rate_syst",False)
        )
        self.out_dir         = kwargs.pop("out_dir",".")
        self.var_lst         = kwargs.pop("var_lst",[])
        self.do_mc_stat      = kwargs.pop("do_mc_stat",False)
        self.coeffs          = kwargs.pop("wcs",[])
        self.use_real_data   = kwargs.pop("unblind",False)
        self.verbose         = kwargs.pop("verbose",True)
        self.use_AAC          = kwargs.pop("use_AAC",False)
        self.wc_scalings     = kwargs.pop("wc_scalings",[])
        self.scalings        = []
        self.use_run3_systs  =  True
        self.suffix          = "_run3"

        # get wc ranges from json
        with open(topeft_path("params/wc_ranges.json"), "r") as wc_ranges_json:
            self.wc_ranges = json.load(wc_ranges_json)

        if self.year_lst:
            for yr in self.year_lst:
                if yr.startswith("UL"):
                    self.use_run3_systs = False
                    self.suffix = "_run2"
                if not yr in self.YEARS:
                    raise ValueError(f"Invalid year choice '{yr}', should be empty if running over all years or one of: {self.YEARS}")
       
        if self.use_run3_systs:
            rate_syst_path = kwargs.pop("rate_systs_path","params/rate_systs_run3.json")
        else:
            rate_syst_path = kwargs.pop("rate_systs_path","params/rate_systs_run2.json")
        explicit_missing_parton_path = kwargs.pop("missing_parton_path",None)
        self.sr_registry, _ = load_or_validate_selected_registry(
            kwargs.pop("sr_registry", DEFAULT_SR_REGISTRY)
        )
        self.missing_parton_payload_path = None
        if self.do_nuisance and not self.skip_missing_parton_rate_syst:
            self.missing_parton_payload_path = self.resolve_missing_parton_payload_path(
                self.year_lst,
                explicit_missing_parton_path,
                self.sr_registry,
            )

        # TODO: Need to find a better name for this variable
        self.rate_systs = self.load_systematics(
            rate_syst_path,
            self.missing_parton_payload_path,
        )

        # Samples to be excluded from the datacard, should correspond to names before group_processes is run
        self.ignore = [
            "DYJetsToLL", "DY10to50", "DY50",
            "ST_antitop_t-channel", "ST_top_s-channel", "ST_top_t-channel", "tbarW", "tW",
            "TTJets",
            "WJetsToLNu",
            # from run3
            "ST_tbarW_Leptonic",
            "ST_tbarW_Semileptonic",
            "TTtoLNu2Q",
            "TTto2L2Nu",
            "ST_tW_Leptonic",
            "ST_tW_Semileptonic",
            "ZG_MLL-50_PTG-200to400", # -->check to see if should put in GROUP
            "ZG_MLL-50_PTG-100to200",
            "ZG_MLL-50_PTG-600",
            "ZG_MLL-4to50_PTG-10to100",
            "ZG_MLL-50_PTG-10to100",
            "ZG_MLL-4to50_PTG-200",
            "ZG_MLL-4to50_PTG-100to200",
            "ZG_MLL-50_PTG-400to600", # --> up to here
            "DYJetsToLL_MLL-50",
            # "TTGamma",
            # "WWTo2L2Nu","ZZTo4L",#"WZTo3LNu",
            # "WWW","WWW_4F","WWZ_4F","WWZ","WZZ","ZZZ",
            # "flips","nonprompt",
            # "tttt","ttlnuJet","tllq","tHq","ttHJet",
            # "TTTo2L2Nu", "TTToSemiLeptonic",
            # "data",
        ]

        if not self.use_real_data:
            # Since we're just going to generate Asimov data, this lets us drop the real data histograms
            #   from the histograms for a minor speed-up
            self.ignore.append("data")

        extra_ignore = kwargs.pop("ignore",[])

        # For now, we leave this as a hardcoded thing, a bit tedious but it works
        # Note: If not explicitly listed, it is assumed that all years should be uncorrelated
        # Note: It is important to list the correlations for ALL years in which it is relevant, so
        #       for example, if a systematic is correlated in 2016, 2016APV, and 2017, there needs
        #       to be an entry for all three years and the list corresponding to each entry needs to
        #       be consistent (i.e. contain the other correlated years) across all three entries
        # Note: As a final note, the actual systematic that appears in the datacards will be just one
        #       of the set to be correlated. So for example, if a systematic is correlated over 2016
        #       and 2016APV, then either the 2016 or 2016APV version will appear in the datacard, but
        #       not both. Typically, the one that remains will be the 2016 version as that's the one
        #       that gets handled first in the loop, but it would be different if we processed things
        #       in a different order

        self.syst_year_corr = {
            "FFcloseEl": {
                "2016": ["2016APV"],
                "2016APV": ["2016"],
                "2022": ["2022EE"],
                "2022EE": ["2022"],
                "2023": ["2023BPix"],
                "2023BPix": ["2023"],
            },
            "FFcloseMu": {
                "2016": ["2016APV"],
                "2016APV": ["2016"],
                "2022": ["2022EE"],
                "2022EE": ["2022"],
                "2023": ["2023BPix"],
                "2023BPix": ["2023"],
            },
        }


        # Defines which systematics should be decorrelated in the self.analysis() step. Each key
        #   should match (exactly) a particular systematic. The list for each systematic specifies
        #   which processes should remain remain correlated or not.
        # Note: A given process should appear AT MOST once in the "matches" list for a given systematic
        #       grouping. If a process has an associated systematic, but doesn't match any of the
        #       groups, then it will retain its original systematic name (i.e. all unmatched
        #       processes will remain correlated).
        # Note: For the special case where the group name is an empty string the systematic will
        #       instead have the matched process' name appended to it, meaning that all matched
        #       processes will be decorrelated!
        # Note: Since the decorrelation happens during the self.analysis() step, the matched names
        #       should correspond to the renamed/re-grouped processes, e.g. use "Diboson" instead of
        #       "ZZ","WZ","WW".
        self.syst_shape_decorrelate = {
            "ISR": [
                {
                    "matches": ["ttH","ttll","tttt","convs"],
                    "group": "gg",
                },
                {
                    "matches": ["ttlnu","tllq","Diboson","Triboson"],
                    "group": "qq",
                },
                {
                    "matches": ["tHq"],
                    "group": "qg"
                }
            ],
            "renorm": [{
                "matches": [".*"],
                "group": "",
            }],
            "fact": [{
                "matches": [".*"],
                "group": "",
            }]
        }

        self.run_decorrelate = ["FES","FF", "FFeta", "FFpt", "btagSFbc_corr", "btagSFlight_corr", "charge_flips", "TES", "lepSF_elec", "lepSF_muon", "lepSF_taus_fake", "lepSF_taus_real"]
        if extra_ignore:
            print(f"Adding processes to ignore: {extra_ignore}")
        self.ignore.extend(extra_ignore)

        self.tolerance = 1e-4
        self.hists = None

        if hists is not None and pkl_path is not None:
            raise ValueError("Specify either 'pkl_path' or 'hists', not both.")
        if hists is None and pkl_path is None:
            raise ValueError("Must specify either 'pkl_path' or 'hists'.")

        tic = time.time()
        self.read(fpath=pkl_path, hists=hists)
        dt = time.time() - tic
        print(f"Total Read+Prune Time: {dt:.2f} s")

        print (f"Saving output to {os.path.realpath(self.out_dir)}")

    def read(self,fpath=None,hists=None):
        """
            Input can be either:
              - a file path to a pkl file containing histograms produced by the topeft.py processor
              - a pre-loaded dictionary of histograms
            The histograms are then pre-processed to remove / group / scale various sparse axes
            categories.
        """
        if hists is not None:
            if not isinstance(hists, dict):
                raise TypeError(
                    f"Expected 'hists' to be a dict, got {type(hists)}"
                )
            self.hists = hists
            merge_report = None
            print(f"Using preloaded histogram dictionary ({len(self.hists)} keys)")
        elif fpath is not None:
            print(f"Opening: {fpath}")
            tic = time.time()
            self.hists, merge_report = load_and_merge_histogram_pkls(
                [fpath],
                require_sumw2=False,
            )
            dt = time.time() - tic
            print(f"Pkl Open Time: {dt:.2f} s")
        else:
            raise ValueError("Need either fpath or hists for read().")

        if is_split_nominal_mapping(self.hists):
            if merge_report is not None and merge_report.get(
                "runtime_histogram_families"
            ):
                runtime_families = merge_report["runtime_histogram_families"]
            else:
                runtime_families = [
                    family
                    for family in list(axes_info) + list(axes_info_2d)
                    if (
                        family in self.hists
                        or f"{family}__scalar_nominal" in self.hists
                        or f"{family}__eft_nominal" in self.hists
                    )
                ]
            self.hists = materialize_legacy_histogram_dict(
                self.hists,
                runtime_families=runtime_families,
                schema_version=NOMINAL_CONTAINER_SCHEMA_VERSION,
            )
            if is_split_nominal_mapping(self.hists):
                raise RuntimeError(
                    "DatacardMaker transient materialization retained split source keys."
                )

        for km_dist, h in self.hists.items():
            if h.empty():
                continue
            km_dist_missing = km_dist not in self.var_lst
            km_base_dist_missing = km_dist.replace('_sumw2','') not in self.var_lst
            if self.var_lst and km_dist_missing and km_base_dist_missing:
                continue
            print(f"Loading: {km_dist}")
            # Remove processes that we don't include in the datacard
            to_remove = []
            for x in h.axes["process"]:
                p = self.get_process(x)
                if p in self.ignore:
                    if self.verbose:
                        print(f"Skipping (ignored): {x}")
                    to_remove.append(x)
                    continue
                if self.year_lst:
                    yr = self.get_year(x)
                    if yr not in self.year_lst:
                        if self.verbose:
                            print(f"Skipping (year): {x}")
                        to_remove.append(x)
                        continue
            h = h.remove("process", to_remove)

            if not self.do_nuisance:
                # Remove all shape systematics
                h.prune("systematic", ["nominal"])

            if self.drop_syst:
                to_drop = set()
                for syst in self.drop_syst:
                    if syst.endswith("Up"):
                        to_drop.add(syst)
                    elif syst.endswith("Down"):
                        to_drop.add(syst)
                    else:
                        to_drop.add(f"{syst}Up")
                        to_drop.add(f"{syst}Down")
                for x in to_drop:
                    print(f"Removing systematic: {x}")
                h = h.remove("systematic", list(to_drop))

            # Remove 'central', 'private', '_4F' text from process names
            grp_map = {}
            for x in h.axes["process"]:
                new_name = (
                    x.replace("private", "").replace("central", "").replace("_4F", "")
                )
                grp_map[new_name] = x
            h = h.group("process", grp_map)

            h = self.group_processes(h)
            h = self.correlate_years(h)

            num_systs = len(h.axes["systematic"])
            print(f"Num. Systematics: {num_systs}")

            self.hists[km_dist] = h

    def channels(self, km_dist):
        return list(self.hists[km_dist].axes["channel"])

    def processes(self, km_dist):
        return list(self.hists[km_dist].axes["process"])

    def binning_view(self, histogram, km_dist, channel):
        """Return the one card-facing histogram view for the selected binning mode."""
        if self.binning_mode == "processing":
            return histogram
        return resolve_and_rebin_histogram(
            histogram,
            km_dist,
            mode="fitting",
            channel=channel,
        )

    def _scaling_histogram_for_json(self, channel_hist, channel, process):
        """Project one card-facing signal histogram onto its scaling payload axes."""
        scaling_hist = channel_hist[
            {
                "channel": channel,
                "process": process,
                "systematic": "nominal",
            }
        ]
        scaling_hist = self.select_final_sr_appl(
            scaling_hist, channel, process=process
        )
        retained_categories = tuple(scaling_hist.categorical_axes.name)
        if retained_categories:
            raise ValueError(
                "Scaling JSON requires category-projected HistEFT input; "
                f"retained categorical axes: {retained_categories!r}."
            )
        return scaling_hist

    # TODO: Can be a static member function
    def load_systematics(self,rs_fpath,mp_fpath):
        """
            Parse out the correlated and decorrelated systematics from rate_systs.json and
            missing_parton.root files.
        """
        rate_systs = {}
        if not self.do_nuisance:
            return rate_systs
        fpath = topeft_path(rs_fpath)
        print(f"Opening: {fpath}")
        with open(fpath) as f:
            rates_json = json.load(f)
        for k1,v1 in rates_json["rate_uncertainties"].items():
            # k1 will be the name of a rate systematic, like 'lumi' or 'pdf_scale'
            if isinstance(v1,dict):
                # This is a correlated rate systematic
                syst_name = f"{k1}"
                new_syst = RateSystematic(syst_name)
                for k2,v2 in v1.items():
                    # k2 will be the name of process, like 'charge_flips' or 'ttH' or 'Diboson'
                    p = self.get_process(k2)
                    new_syst.add_process(p,v2)
                rate_systs[k1] = new_syst
            else:
                # The systematic gets applied to everything
                syst_name = f"{k1}"
                new_syst = RateSystematic(syst_name,all=True,unc=v1)
                rate_systs[k1] = new_syst

        # Certain rate systematics are only correlated between subsets of processes
        to_remove = set()
        for p,corr_systs in rates_json["correlations"].items():
            # 'p' is the name of a process and 'corr_systs' is a dictionary defining which rate systematic
            #   needs to be decorrelated into a specific sub-group, e.g. for ttH: pdf_scale -> pdf_scale_gg
            for syst,grp in corr_systs.items():
                # 'syst' should be the name of a systematic already defined in the 'rate_systs' dictionary
                #   and 'grp' is the string we are going to differentiate it from the other variants of 'syst'
                to_remove.add(syst)
                syst_name = f"{syst}_{grp}"
                if not syst_name in rate_systs:
                    rate_systs[syst_name] = RateSystematic(syst_name)
                if not rate_systs[syst].has_process(p):
                    print(f"Warning: No process {p} found for {syst} systematic")
                    continue
                unc = rate_systs[syst].get_process(p)
                rate_systs[syst_name].add_process(p,unc)
        # Now lets remove the original systematics which we decorrelated into sub-groups
        for syst in to_remove:
            rate_systs.pop(syst)

        # Note: The 'diboson_njets' and 'missing_parton' uncertainties are a bit special, the values we
        #   store in their corresponding RateSystematic objects will be dictionaries that encode the
        #   uncertainty split by njets

        # Now deal with the 'diboson_njets' systematic for Dibosons
        syst_name = "diboson_njets"
        # new_syst = RateSystematic(syst_name)
        new_syst = JetScale(syst_name)
        for p,per_jet_uncs in rates_json["diboson_njets"].items():
            new_syst.add_process(p,per_jet_uncs)
        rate_systs[syst_name] = new_syst

        if getattr(self, "skip_missing_parton_rate_syst", False):
            print("Skipping missing_parton rate systematic")
        else:
            # Finally, deal with the missing_parton systematic
            # TODO: This feels pretty hardcoded, but not sure there's any way around it
            syst_name = self.missing_parton_nuisance_name_for_years(
                self.year_lst,
                payload_path=mp_fpath,
            )
            new_syst = RateSystematic(syst_name)

            fpath = topeft_path(mp_fpath)
            print(f"Opening: {fpath}")
            payload_values = validate_legacy_missing_parton_payload(
                fpath,
                sr_registry=self.sr_registry,
            )
            # Values in the ROOT file are fractional rate shifts, so add 1 to
            # obtain the corresponding kappa values used by the datacard.
            d = {
                channel_key: values + 1
                for channel_key, values in payload_values.items()
            }
            new_syst.add_process("tllq",d)
            new_syst.add_process("tHq",d)
            rate_systs[syst_name] = new_syst

        return rate_systs

    # TODO: Can be a static member function
    def group_processes(self,h):
        """
            Groups together certain processes from the 'process' axis. We also abuse this method to
            rename specific process categories. Both of which are determined by the GROUP static data
            member.
        """
        # TODO: This needs work to be less convoluted...
        all_procs = set(h.axes["process"])
        grp_map = {}
        for grp_name,to_grp in self.GROUP.items():
            for yr in self.YEARS:
                new_name = f"{grp_name}{yr}"
                lst = []
                for x in to_grp:
                    old_name = f"{x}{yr}"
                    if old_name in all_procs:
                        lst.append(old_name)
                        all_procs.remove(old_name)
                # Note: Some processes only exist in certain channels (e.g. flips), so we need to
                #   skip them when they don't appear in the identifiers list
                if len(lst):
                    grp_map[new_name] = lst
        # Include back in everything that wasn't specified by the initial groupings
        for x in all_procs:
            grp_map[x] = [x]
        h = h.group("process", grp_map)
        return h

    # TODO: Can be a static member function
    def correlate_years(self,h):
        """
            Merges together different run years, taking care to treat year-specific systematics as
            uncorrelated from one another
        """
        if not self.do_nuisance:
            # Only sum over the years, don't mess with nuisance stuff
            grp_map = defaultdict(lambda: [])
            for x in h.axes["process"]:
                p = self.get_process(x)
                grp_map[p].append(x)
            h = h.group("process", grp_map)
            return h
        # This requires some fancy footwork to make work
        print("Correlating years")

        # Need to figure out which years are actually present in the histogram
        unique_proc_years = set()
        for x in h.axes["process"]:
            yr = self.get_year(x)
            unique_proc_years.add(yr)

        already_correlated = set()  # Keeps track of which systematics have already been correlated
        for sp_key in h.categorical_keys:
            proc = sp_key.process
            syst = sp_key.systematic
            proc_year = self.get_year(proc)
            syst_year = self.get_year(syst)
            if syst_year is None:
                # This ensures that the systematic in question is a per-year systematic
                continue
            if syst in already_correlated:
                if self.verbose:
                    print(f"Skipping {syst} as it was already correlated in a previous year")
                continue
            syst_base = self.strip_fluctuation(syst)
            syst_base = self.strip_year(syst_base)
            corr_keys = []
            for p_yr, s_yr in zip(self.YEARS, self.SYST_YEARS): 
                if p_yr not in unique_proc_years:
                    # The histogram file was generated by running over a subset of the years or we are
                    # only making cards for a certain year
                    continue
                if p_yr == proc_year:
                    # We never add self to self
                    continue
                if syst_base in self.syst_year_corr and s_yr in self.syst_year_corr[syst_base] and syst_year in self.syst_year_corr[syst_base][s_yr]:
                    # The systematic for this year needs to be correlated
                    syst_key = syst.replace(syst_year, s_yr)
                    already_correlated.add(syst_key)
                else:
                    # The systematic for this year needs to be uncorrelated
                    syst_key = "nominal"
                proc_key = proc.replace(proc_year, p_yr)

                # Construct the sparse key
                corr_key = sp_key._asdict()
                corr_key["process"] = proc_key
                corr_key["systematic"] = syst_key
                corr_key = type(sp_key)(**corr_key)
                corr_keys.append(corr_key)

            for k in corr_keys:
                h[sp_key] += h[k]

            sp_tup = tuple(sp_key)
            if self.verbose:
                print(f"{tuple(sp_tup)} -- {' + '.join(map(lambda k: str(tuple(k)), corr_keys))}")

        # Finally sum over years, since the per-year systematics only appear in a corresponding
        #   "process year", the grouping for those systematics just adds itself with nothing from
        #   the other process years
        grp_map = defaultdict(lambda: [])
        for x in h.axes["process"]:
            p = self.get_process(x)
            grp_map[p].append(x)
        h = h.group("process", grp_map)

        # Remove the categories which were already correlated together so as to not double count
        if already_correlated:
            for k in already_correlated:
                if self.verbose: print(f"Removing: {k}")
            h = h.remove("systematic", list(already_correlated))

        return h

    def get_selected_wcs(self,km_dist,ch_lst=[]):
        """
            For each process, iterates over every channel and every bin checking the EFT parameterization
            coefficients for if they have a significant impact or not relative to the SM contribution. If
            any term from any channel+bin is determined to be significant, the WC is selected, otherwise
            it is excluded for that process and won't be included in the EFT decomposition
        """
        tic = time.time()
        h = self.hists[km_dist].integrate("systematic",["nominal"])
        channels = list(h.axes["channel"])
        if ch_lst:
            if self.verbose:
                print(f"Selecting WCs from subset of channels: {ch_lst}")
            requested = set(ch_lst)
            channels = [channel for channel in channels if channel in requested]

        procs = list(h.axes["process"])
        selected_wcs = {p: set() for p in procs}

        wcs = ["sm"] + h.wc_names

        # This maps a WC to a list whose elements are the indices of the coefficient array of the
        #   HistEFT that involve that particular WC
        # NOTE: Building up the index array MUST match exactly with how the HistEFT coeff array is
        #       constructed/computed [1], otherwise the index array that gets computed won't pick
        #       out the correct coeff array indices for the corresponding WC!
        # [1] https://github.com/TopEFT/topcoffea/blob/3bef686fead216183ebb6dfb464e67629cfe75f5/topcoffea/modules/eft_helper.py#L32-L36
        wc_to_terms = {}
        start_index = 1
        index = start_index
        for i in range(len(wcs)):
            wc1 = wcs[i]
            wc_to_terms[wc1] = set()
            for j in range(i+1):
                wc2 = wcs[j]
                wc_to_terms[wc1].add(index)
                wc_to_terms[wc2].add(index)
                index += 1

        # Convert the set to a sorted np.array
        for wc in wcs:
            wc_to_terms[wc] = np.array(sorted(wc_to_terms[wc]))

        for p in procs:
            if not self.is_signal(p):
                continue
            for wc,idx_arr in wc_to_terms.items():
                if len(self.coeffs) and not wc in self.coeffs:
                    continue
                if wc == "sm":
                    continue
                if wc == "ctlTi" and p == "tttt":
                    continue
                selected = False
                for channel in channels:
                    channel_hist = self.binning_view(
                        h.integrate("channel", [channel]), km_dist, channel
                    )
                    p_hist = channel_hist.integrate("process", [p])
                    for sp_key, arr in p_hist.view(as_dict=True, flow=True).items():
                        # Ignore underflow and overflow bins. The remaining bins
                        # are the exact fitting bins used by the card templates.
                        sl_arr = arr[1:-1]
                        sm_norm = np.where(
                            sl_arr[:,start_index] < 1e-5,
                            999,
                            sl_arr[:,start_index],
                        )
                        n_arr = (sl_arr.T / sm_norm).T
                        wc_terms = np.abs(n_arr[:,idx_arr])
                        if np.any(wc_terms > self.tolerance):
                            selected_wcs[p].add(wc)
                            selected = True
                            break
                    if selected:
                        break
        if self.verbose:
            dt = time.time() - tic
            print(f"WC Selection Time: {dt:.2f} s")
        return selected_wcs

    def make_scalings_json(self,scalings_json,ch,km_dist,p,wc_names,scalings):
        scalings = scalings.tolist()
        scalings_json.append(
            {
                "channel": ch + "_" + str(km_dist),
                "process": p + "_sm",  # NOTE: needs to be in the datacard
                "parameters": ["cSM[1]"] + [
                    self.format_wc(wcname) for wcname in wc_names
                ],
                "scaling":
                    scalings[1:], # exclude underflow bin
            }
        )
        return scalings_json

    def format_wc(self,wcname):
        lo, hi = self.wc_ranges[wcname]
        return "%s[0,%.1f,%.1f]" % (wcname, lo, hi)

    def analyze(self,km_dist,ch,selected_wcs, crop_negative_bins, wcs_dict):
        """ Handles the EFT decomposition and the actual writing of the ROOT and text datacard files."""
        if not km_dist in self.hists:
            print(f"[ERROR] Unknown kinematic distribution: {km_dist}")
            return None
        elif ch not in self.hists[km_dist].axes["channel"]:
            print(f"[ERROR] Unknown channel {ch}")
            return None

        h = self.hists[km_dist]
        h_sumw2 = self.hists.get(f"{km_dist}_sumw2")
        self._resolve_supported_sr_appl(h, ch)
        self._resolve_supported_sr_appl(h_sumw2, ch)

        print(f"Analyzing {km_dist} in {ch}")

        bin_str = f"bin_{ch}_{km_dist}"
        col_width = max(PRECISION*2+5,len(bin_str))
        syst_width = 0

        if km_dist != "njets":
            num_j,num_b = self.get_jet_mults(ch)
        else:
            num_j,num_b = 0,0
        num_l = self.get_lep_mult(ch)
        if num_l == 2 or num_l == 4:
            num_b = 2

        outf_root_name = self.FNAME_TEMPLATE.format(cat=ch,kmvar=km_dist,ext="root")

        if h_sumw2 is None:
            msg = "No sumw2 histogram found! Setting errors to 0"
            print(msg)
        ch_hist = self.binning_view(h.integrate("channel",[ch]), km_dist, ch)
        ch_sumw2 = (
            None
            if h_sumw2 is None
            else self.binning_view(h_sumw2.integrate("channel",[ch]), km_dist, ch)
        )
        if ch_sumw2 is not None:
            validate_matching_histogram_edges(
                ch_hist,
                ch_sumw2,
                context=f"datacard nominal/sumw2 for {km_dist}:{ch}",
            )
        data_obs = np.zeros((2, ch_hist.dense_axis.extent))

        print(f"Generating root file: {outf_root_name}")
        tic = time.time()
        num_h = 0
        all_shapes = set()
        text_card_info = {}
        outf_root_name = os.path.join(self.out_dir,outf_root_name)
        with uproot.recreate(outf_root_name) as f:
            for p,wcs in selected_wcs.items():
                # TODO This is a hack for now, track this upstream
                if 'flip' in p and '2l' not in ch:
                    continue
                # TODO This is a hack for now, track this upstream
                if 'fakes' in p and '4l' in ch:
                    continue
                if 'nonprompt' in p and '4l' in ch:
                    continue
                if p == "fakes" and h_sumw2 is None:
                    raise RuntimeError(
                        f"DatacardMaker requires '{km_dist}_sumw2' for the final fakes process."
                    )
                if 'flip' in p and '2los' in ch:
                    continue

                proc_hist = ch_hist.integrate("process",[p])
                if ch_sumw2 is None:
                    proc_sumw2 = None
                elif p in ch_sumw2.axes["process"]:
                    proc_sumw2 = ch_sumw2.integrate("process",[p])
                elif process_retains_stat_uncertainty(p):
                    raise RuntimeError(
                        "DatacardMaker requires a process companion for "
                        f"{p!r} in '{km_dist}_sumw2' because it retains stored "
                        "statistical uncertainty."
                    )
                else:
                    proc_sumw2 = None
                proc_hist = self.select_final_sr_appl(proc_hist, ch, process=p)
                proc_sumw2 = self.select_final_sr_appl(proc_sumw2, ch, process=p)
                if self.verbose:
                    print(f"Decomposing {ch}-{p}")
                decomposed_templates = self.decompose(proc_hist,proc_sumw2,wcs)
                is_eft = self.is_signal(p)
                # Note: This feels like a messy way of picking out the data_obs info
                if p == "data":
                    data_sm = decomposed_templates.pop("sm")
                    if self.use_real_data:
                        if len(data_sm) != 1:
                            raise RuntimeError("obs data has unexpected number of sparse bins")
                        elif sum(data_obs[0]) != 0:
                            raise RuntimeError("filling obs data more than once!")
                        for sp_key,arr in data_sm.items():
                            data_obs += arr
                if not self.use_AAC:
                    decomposed_templates = {k: v for k, v in decomposed_templates.items() if k == 'sm'}
                for base,v in decomposed_templates.items():
                    proc_name = f"{p}_{base}"
                    col_width = max(len(proc_name),col_width)
                    text_card_info[proc_name] = {
                        "shapes": set(),
                        "rate": -1
                    }
                    self.validate_sparse_axes_for_card(v, ch, proc_name)
                    if crop_negative_bins:
                        v = _sanitize_negative_template_bins(v)
                    _validate_ff_template_support(
                        v,
                        variable=km_dist,
                        channel=ch,
                        process=p,
                        decomposition=base,
                    )
                    # There should be only 1 sparse axis at this point, the systematics axis
                    seen = {}
                    written_hist_names = set()
                    for sp_key,arr in v.items():
                        syst = sp_key[0]
                        syst = sp_key.systematic

                        syst_base, syst = resolve_shape_nuisance_identity(
                            syst, self.suffix, self.run_decorrelate
                        )

                        if syst_base == "JES_Total":
                            continue
                        sum_arr = sum(arr[0])
                        if sum_arr == 0: continue #TODO find a more elegant solution

                        if syst_base not in seen:
                            seen[syst_base] = [False, False] # check[Up, Down]
                        if "Up" in syst:
                            seen[syst_base][0] = True
                        if "Down" in syst:
                            seen[syst_base][1] = True

                        if syst == "nominal" and base == "sm":
                            if self.verbose:
                                print(f"\t{proc_name:<12}: {sum_arr:.4f} {arr[0]}")
                            if not self.use_real_data:
                                # Create asimov dataset
                                vals = wcs_dict # set wcs to certain values from command line
                                decomposed_templates_Asimov = self.decompose(proc_hist,proc_sumw2,wcs,vals)
                                data_sm = decomposed_templates_Asimov.pop("sm")
                                data_obs += data_sm[sp_key]
                        if syst == "nominal":
                            hist_name = f"{proc_name}"
                            text_card_info[proc_name]["rate"] = sum_arr
                        else:
                            hist_name = f"{proc_name}_{syst}"
                            # Systematics in the text datacard don't have the Up/Down postfix
                            #syst_base = syst.replace("Up","").replace("Down","")
                            if syst_base in self.syst_shape_decorrelate:
                                # We want to split this systematic to be uncorrelated between certain
                                #   processes, so we modify the systematic name to make combine treat
                                #   them as separate systematics. Also, we use 'p' instead of 'proc_name'
                                #   for renaming since we want the decomposed EFT terms for a particular
                                #   process to share the same nuisance parameter
                                matched = []
                                for r in self.syst_shape_decorrelate[syst_base]:
                                    if regex_match([p],r["matches"]):
                                        # The matched process should have this systematic put into a new group
                                        matched.append(r["group"])
                                if len(matched) == 0:
                                    # No matches found, so keep the original systematic name
                                    split_syst = syst_base
                                elif len(matched) == 1:
                                    # Found a match, so decorrelate the process from non-matched processes
                                    group = matched[0]
                                    split_syst = f"{syst_base}_{group}"
                                    if group == "":
                                        # In the special case that the group is an empty string,
                                        #   decorrelate ALL matched processes
                                        split_syst = f"{syst_base}_{p}"
                                else:
                                    # We shouldn't have more than one match for a given systematic
                                    raise RuntimeError(f"Unable to decorrelate shape systematic {syst_base} for {p}. Multiple group matches found: {matched}")
                                hist_name = hist_name.replace(syst_base,split_syst)
                                all_shapes.add(split_syst)
                                if seen[syst_base] == [True, True]:
                                    text_card_info[proc_name]["shapes"].add(split_syst)
                                if base == "sm" and self.verbose:
                                    print(f"\tDecorrelate {p} for {syst_base} into {split_syst} ({syst.replace(syst_base,'')})")
                            else:
                                all_shapes.add(syst_base)
                                if seen[syst_base] == [True, True]:
                                    text_card_info[proc_name]["shapes"].add(syst_base)
                            syst_width = max(len(syst),syst_width)
                        zero_out_sumw2 = not process_retains_stat_uncertainty(p)
                        if hist_name in written_hist_names:
                            raise ValueError(
                                f"Duplicate ROOT template name {hist_name!r} while writing "
                                f"datacard channel {ch!r}, process {proc_name!r}. An "
                                "unexpected sparse axis was not resolved before template "
                                "writing."
                            )
                        written_hist_names.add(hist_name)
                        f[hist_name] = to_hist(arr,hist_name,zero_wgts=zero_out_sumw2)

                        num_h += 1
                    if km_dist == "njets":
                        # We need to handle certain systematics differently when looking at njets procs
                        if p == "Diboson":
                            # Handle the 'diboson_njets' uncertainty
                            # syst = "diboson_njets"
                            # hist_name = f"{proc_name}_{syst}"
                            # syst_kappa = self.rate_systs[syst].get_process(p)[str(num_j)]
                            # if syst_kappa == "-":
                            #     raise ValueError(f"The kappa value for {syst} is missing!")
                            pass

                        if p == "tllq" or p == "tHq":
                            # Handle the 'missing_parton' uncertainty
                            pass
                # obtain the scalings for scalings.json file
                if p in self.SIGNALS:
                    scaling_hist = self._scaling_histogram_for_json(ch_hist, ch, p)
                    validate_matching_histogram_edges(
                        proc_hist,
                        scaling_hist,
                        context=f"datacard template/scaling for {km_dist}:{ch}:{p}",
                    )
                    if self.wc_scalings:
                        scalings = scaling_hist.make_scaling(wc_list=self.wc_scalings)
                        self.scalings_json = self.make_scalings_json(self.scalings,ch,km_dist,p,self.wc_scalings,scalings)
                    else:
                        scalings = scaling_hist.make_scaling()
                        self.scalings_json = self.make_scalings_json(self.scalings,ch,km_dist,p,h.wc_names,scalings)
            f["data_obs"] = to_hist(data_obs,"data_obs")

        line_break = "##----------------------------------\n"
        left_width = len(line_break) + 2
        left_width = max(syst_width+len("shape")+1,left_width)

        outf_card_name = self.FNAME_TEMPLATE.format(cat=ch,kmvar=km_dist,ext="txt")
        print(f"Generating text file: {outf_card_name}")
        outf_card_name = os.path.join(self.out_dir,outf_card_name)
        with open(outf_card_name,"w") as f:
            f.write(f"shapes *        * {os.path.split(outf_root_name)[1]} $PROCESS $PROCESS_$SYSTEMATIC\n")
            f.write(line_break)
            f.write(f"bin         {bin_str}\n")
            f.write(f"observation {sum(data_obs[0]):.{PRECISION}f}\n")
            f.write(line_break)
            f.write(line_break)

            # Note: This list is what controls the columns in the text datacard, if a process appears
            #       in this list it should NEVER be skipped in any of the following for loops.
            # proc_order = sorted(text_card_info.keys())
            proc_order = [k for k in text_card_info.keys() if text_card_info[k]["rate"] != -1]  # rate = -1 only happens when there's no syst histograms (e.g. flips in 3l/4l)

            # Bin row
            row = [f"{'bin':<{left_width}}"]    # Python string formatting is pretty great!
            for p in proc_order:
                row.append(f"{bin_str:>{col_width}}")
            row = " ".join(row) + "\n"
            f.write(row)

            # 1st process row
            row = [f"{'process':<{left_width}}"]
            for p in proc_order:
                row.append(f"{p:>{col_width}}")
            row = " ".join(row) + "\n"
            f.write(row)

            # 2nd process row
            row = [f"{'process':<{left_width}}"]
            bkgd_count =  1
            sgnl_count = -1
            for p in proc_order:
                if any([x in p for x in self.SIGNALS]): # Check for if the process is signal or not
                    row.append(f"{sgnl_count:>{col_width}}")
                    sgnl_count += -1
                else:
                    row.append(f"{bkgd_count:>{col_width}}")
                    bkgd_count += 1
            row = " ".join(row) + "\n"
            f.write(row)

            # Rate row
            row = [f"{'rate':<{left_width}}"]
            for p in proc_order:
                r = text_card_info[p]["rate"]
                if r < 0:
                    print(f"Process {p} has negative total rate: {r:.3f} -> setting to 0 in text card")
                    r = 0
                row.append(f"{r:>{col_width}.{PRECISION}f}") # Do not challenge me on Python string formatting!
            row = " ".join(row) + "\n"
            f.write(row)
            f.write(line_break)

            # Shape systematics rows
            for syst in sorted(all_shapes):
                left_text = f"{syst:<{syst_width}} shape"
                row = [f"{left_text:<{left_width}}"]
                for p in proc_order:
                    if syst in text_card_info[p]["shapes"]:
                        row.append(f"{'1':>{col_width}}")
                    else:
                        row.append(f"{'-':>{col_width}}")
                row = " ".join(row) + "\n"
                f.write(row)

            # Rate systematics rows
            for k,rate_syst in self.rate_systs.items():
                syst_name = rate_syst.name
                left_text = f"{syst_name:<{syst_width}} lnN"
                if km_dist == "njets" and (
                    syst_name == "diboson_njets" or self.is_missing_parton_nuisance_name(syst_name)
                ):
                    # These systematics are only treated as rate systs for njets distribution
                    continue
                row = [f"{left_text:<{left_width}}"]
                for p in proc_order:
                    proc_name = self.get_process(p) # Strips off any "_sm" or "_lin_*" junk
                    # Need to handle certain systematics in a special way
                    if syst_name == "diboson_njets":
                        v = rate_syst.get_process(proc_name,num_j)
                        # v = rate_syst.get_process(proc_name)
                        # if isinstance(v,dict):
                        #     v = v[str(num_j)]
                    elif self.is_missing_parton_nuisance_name(syst_name):
                        v = rate_syst.get_process(proc_name)
                        #if miss_part_path == "data/missing_parton/missing_parton.root":
                        #    if "2los" in ch:
                        #        ch = ch.replace("2los", "2lss").replace("_onZ", "_p")
                        #    # First strip off any njet and/or bjet labels
                        #    ch_key = ch.replace(f"_{num_j}j","").replace(f"_{num_b}b","").replace("_1tau", "")
                        ## Now construct the category key, matching names in the missing_parton file to the current category
                        #    if num_l == 2:
                        #        njet_offset = 4
                        #        ch_key = ch_key.replace("_onZ", "")
                        #        ch_key = ch_key.replace("_offZ", "")
                        #        ch_key = f"{ch_key}_{num_b}b"
                        #    elif num_l == 3:
                        #        njet_offset = 2
                        #        if "_onZ" in ch:
                        #            ch_key = f"{num_l}l_sfz_{num_b}b"
                        #        elif "_p_offZ" in ch:
                        #            ch_key = f"{num_l}l{num_b}b_p"
                        #        elif "_m_offZ" in ch:
                        #            ch_key = f"{num_l}l{num_b}b_m"
                        #        elif "tau" in ch:
                        #            ch_key = f"{num_l}l_sfz_{num_b}b"
                        #        else:
                        #            raise ValueError(f"Unable to match {ch} for {syst_name} rate systematic")
                        #    elif num_l == 4:
                        #        njet_offset = 2
                        #        ch_key = f"{ch_key}_{num_b}b"
                        #    else:
                        #        raise ValueError(f"Unable to match {ch} for {syst_name} rate systematic")
                        #    #The bins in the missing_parton root files start indexing from 0
                        #    bin_idx = num_j - njet_offset
                        #else:
                        ch_key = ch.replace(f"_{num_j}j","")
                        bin_idx = num_j
                        if isinstance(v,dict):
                        
                            # Skip channels that don't exist in missing_parton file
                            if ch_key not in v:
                                v = "-"
                            else:
                                unc_hi = v[ch_key][bin_idx]
                                unc_lo = max(0.01,2 - unc_hi)
                                print("ch", ch_key)
                                print(unc_hi, unc_lo)
                                v = f"{unc_lo:.{PRECISION}f}/{unc_hi:.{PRECISION}f}"


                        elif v != "-":
                            raise ValueError(f"The missing_parton systematic isn't a dictionary (ch={ch}): {v}")
                    else:
                        v = rate_syst.get_process(proc_name)
                    row.append(f"{v:>{col_width}}")
                row = " ".join(row) + "\n"
                f.write(row)

            if self.do_mc_stat:
                f.write("* autoMCStats 10\n")
            else:
                f.write("* autoMCStats -1\n")
        dt = time.time() - tic
        print(f"File Write Time: {dt:.2f} s")
        print(f"Total Hists Written: {num_h}")
        for syst_base, (has_up, has_down) in seen.items():        
            if has_up and not has_down:
                print(f"Missing 'Down' uncertainty in {syst_base}")
        
            elif has_down and not has_up:
                print(f"Missing 'Up' uncertainty in {syst_base}")
        print("check")
    # TODO: Can be a static member function

    def decompose(self,h,sumw2,wcs,vals={}):
        """
            Decomposes the EFT quadratic parameterization coefficients into combinations that result
            in non-negative coefficient terms.

            Note: All other WCs are assumed set to 0
            sm piece:    set(c1=0.0)
            lin piece:   set(c1=1.0)
            mixed piece: set(c1=1.0,c2=1.0)
            quad piece:  0.5*[set(c1=2.0) - 2*set(c1=1.0) + set(sm)]
        """
        tic = time.time()

        sm = h.eval({})
        sm_w2 = None if sumw2 is None else sumw2.eval(vals)
        sm = add_sumw2_stub(sm,sm_w2)

        # Note: The keys of this dictionary are a pretty contrived, but are useful later on
        r = {}
        r["sm"] = sm
        terms = 1
        for n1, wc1 in enumerate(wcs):
            tmp_lin_1 = h.eval({wc1: 1.0})
            tmp_lin_2 = h.eval({wc1: 2.0})

            tmp_lin_1 = add_sumw2_stub(tmp_lin_1)
            tmp_lin_2 = add_sumw2_stub(tmp_lin_2)

            lin_name = f"lin_{wc1}"
            quad_name = f"quad_{wc1}"

            terms += 2

            r[lin_name] = tmp_lin_1
            r[quad_name] = {}
            for sp_key in h.categorical_keys:
                r[quad_name][sp_key] = []
                for i in range(2):
                    r[quad_name][sp_key].append(
                        0.5 * (
                            tmp_lin_2[sp_key][i] - 2 * tmp_lin_1[sp_key][i] + sm[sp_key][i]
                        )
                    )

            for n2, wc2 in enumerate(wcs):
                if n1 >= n2:
                    continue
                mixed_name = f"quad_mixed_{wc1}_{wc2}"
                mixed = h.eval({wc1: 1.0, wc2: 1.0})
                mixed = add_sumw2_stub(mixed)
                r[mixed_name] = mixed
                terms += 1

        toc = time.time()
        dt = toc - tic
        if self.verbose:
            print(f"\tDecompose Time: {dt:.2f} s")
            print(f"\tTotal terms: {terms}")

        return r

if __name__ == '__main__':
    fpath = topeft_path("../analysis/topEFT/histos/may18_fullRun2_withSys_anatest08_np.pkl.gz")

    tic = time.time()
    dc = DatacardMaker(fpath)

    km_dist = "lj0pt"
    chans = ["2lss_m_4j","2lss_4t_m_4j"]
    # km_dist = "njets"
    # chans = ["2lss_m","2lss_4t_m"]

    target_selected = {
        "tHq": ["ctp", "cptb", "cQq13", "cbW", "cpQ3", "ctW", "cQq83", "ctG"],
        "tllq": ["cpt", "cptb", "cQlMi", "cQl3i", "ctlTi", "ctli", "cQq13", "cbW", "cpQM", "cpQ3", "ctei", "cQei", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "ttH": ["cpt", "ctp", "cptb", "cQq81", "cQq11", "ctq8", "ctq1", "cQq13", "cbW", "cpQM", "cpQ3", "ctW", "cQq83", "ctZ", "ctG"],
        "ttll": ["cpt", "cptb", "cQlMi", "cQq81", "cQq11", "cQl3i", "ctq8", "ctlTi", "ctq1", "ctli", "cQq13", "cbW", "cpQM", "cpQ3", "ctei", "cQei", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "ttlnu": ["cpt", "ctp", "cQlMi", "cQq81", "cQq11", "cQl3i", "ctq8", "ctlTi", "ctq1", "ctli", "cQq13", "cpQM", "cpQ3", "ctW", "ctlSi", "cQq83", "ctZ", "ctG"],
        "tttt": ["cpt", "ctp", "cptb", "cQq81", "cQq11", "ctq8", "ctq1", "cQq13", "cbW", "cpQM", "cpQ3", "ctW", "cQq83", "ctZ", "ctG", "ctt1", "cQt1", "cQt8", "cQQ1"]
    }

    selected_wcs = dc.get_selected_wcs(km_dist)
    for p,tar_wcs in target_selected.items():
        if p not in selected_wcs:
            print(f"Skipping {p} for selected WC comparison")
            continue
        sel_wcs = selected_wcs[p]
        print(f"old {p:>5}: {sorted(tar_wcs)}")
        print(f"new {p:>5}: {sorted(sel_wcs)}")
        miss_old = set(tar_wcs).difference(sel_wcs)
        miss_new = sel_wcs.difference(set(tar_wcs))

        print(f"Missing from old: {sorted(miss_old)}")
        print(f"Missing from new: {sorted(miss_new)}")
        print("-"*50)

    for cat in dc.channels(km_dist):
        if not cat in chans:
            continue
        r = dc.analyze(km_dist,cat,selected_wcs, True)
    dt = time.time() - tic
    print(f"Total Time: {dt:.2f} s")

    wc_to_terms = {}
    h = dc.hists[km_dist]
    wcs = ["sm"] + h._wcnames

    index = 0
    for i in range(len(wcs)):
        wc1 = wcs[i]
        wc_to_terms[wc1] = set()
        for j in range(i+1):
            wc2 = wcs[j]
            wc_to_terms[wc1].add(index)
            wc_to_terms[wc2].add(index)
            index += 1

    for wc in wcs:
        terms = sorted(wc_to_terms[wc])
        s1 = ", ".join([f"{x:>3d}" for x in terms[:6]])
        s2 = terms[-1]
        print(f"{wc:>5}: [{s1}, ... , {s2:>3d} ]")
