"""Focused helpers for generated production-profile sidecar fixtures."""

from topeft.modules.data_driven_products import resolve_data_driven_products
from topeft.modules.production_sample_profile import (
    build_active_sample_universe,
    certify_production_sample_contract,
)


def certify_test_profile(policy, samples, products=None):
    if products is None:
        products = resolve_data_driven_products(
            {
                "nonprompt": {"enabled": False},
                "flips": {"enabled": False},
            },
            data_driven_products_present=True,
            legacy_do_np=False,
            samples=samples,
            runtime_families=policy.runtime_histogram_families,
            metadata_path="pytest",
        )
    return certify_production_sample_contract(
        build_active_sample_universe(samples, wrapper_identity="pytest"),
        policy,
        products,
    )
