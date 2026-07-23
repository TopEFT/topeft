"""Deprecated compatibility boundary for the combined renorm/fact envelope."""

import argparse


unsupported_renormfact_envelope_message = (
    "The maintained analysis uses independent renormUp/Down and factUp/Down "
    "shape nuisances. The historical combined renorm/fact envelope is "
    "unsupported and deprecated because correlated renormfact templates and "
    "their normalization are not produced. No histogram or output was modified."
)


def raise_unsupported_renormfact_envelope() -> None:
    """Raise the common fail-closed error for every legacy envelope surface."""

    raise RuntimeError(unsupported_renormfact_envelope_message)


def get_renormfact_envelope(dict_of_hists, *, verbose=True):
    """Retained only for import compatibility; the transformation is unsupported."""

    raise_unsupported_renormfact_envelope()


def apply_renormfact_envelope_to_histogram(histo, *, verbose=True, hist_name=None):
    """Retained only for import compatibility; the transformation is unsupported."""

    raise_unsupported_renormfact_envelope()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deprecated combined renorm/fact-envelope entry point."
    )
    parser.add_argument("pkl_file_path", help="Legacy input path; never opened.")
    parser.add_argument("-n", "--output-name", default="histos_dict")
    parser.parse_args()
    raise_unsupported_renormfact_envelope()


if __name__ == "__main__":
    main()
