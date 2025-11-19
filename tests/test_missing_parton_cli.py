from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MISSING_PARTON_PATH = Path(__file__).resolve().parents[1] / 'analysis' / 'topeft_run2' / 'missing_parton.py'
spec = spec_from_file_location('missing_parton', MISSING_PARTON_PATH)
missing_parton = module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(missing_parton)


def test_default_channels_for_njets():
    channels = missing_parton.determine_files('njets', None)
    assert len(channels) == len(missing_parton.DEFAULT_CHANNELS)
    assert len(channels) > 2


def test_channel_override():
    subset = ['2lss_fwd_m', '2lss_fwd_p']
    assert missing_parton.determine_files('njets', subset) == subset


def test_var_overrides_diff_and_ptz_lists():
    assert missing_parton.determine_files('ht', None) == missing_parton.FILES_DIFF
    assert missing_parton.determine_files('ptz', None) == missing_parton.FILES_PTZ
