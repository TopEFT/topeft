import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
cfg_root = repo / "input_samples/cfgs"
driver = repo / "analysis/topeft_run2/project01_copy_manifest.sh"
project01 = "file:///project01/ndcms/apiccine"
cfgs = ["mc_signal_samples_NDSkim.cfg", "mc_background_samples_NDSkim.cfg", "mc_background_samples_cr_NDSkim.cfg", "data_samples_NDSkim.cfg", "NDSkim_2022_background_samples.cfg", "NDSkim_2022_background_samples_cr.cfg", "NDSkim_2022_data_samples.cfg", "NDSkim_2022_mc_signal_samples.cfg", "NDSkim_2022_mc_signal_samples_sr.cfg", "NDSkim_2022EE_background_samples.cfg", "NDSkim_2022EE_background_samples_cr.cfg", "NDSkim_2022EE_data_samples.cfg", "NDSkim_2022EE_mc_signal_samples.cfg", "NDSkim_2022EE_mc_signal_samples_sr.cfg", "NDSkim_2023_background_samples.cfg", "NDSkim_2023_background_samples_cr.cfg", "NDSkim_2023_data_samples.cfg", "NDSkim_2023_mc_signal_samples.cfg", "NDSkim_2023_mc_signal_samples_sr.cfg", "NDSkim_2023BPix_background_samples.cfg", "NDSkim_2023BPix_background_samples_cr.cfg", "NDSkim_2023BPix_data_samples.cfg", "NDSkim_2023BPix_mc_signal_samples.cfg", "NDSkim_2023BPix_mc_signal_samples_sr.cfg"]

def test_cfgs_have_project01_only_as_effective_prefix():
    assert len(cfgs) == 24
    for name in cfgs:
        prefix = ""; seen = []; text = (cfg_root / name).read_text()
        for line in text.splitlines():
            token = line.split("#", 1)[0].strip().replace(" ", "")
            if token.startswith(("root://", "file://")): prefix = token
            elif token.endswith(".json"): seen.append(prefix)
        assert seen and set(seen) == {project01}
        assert "root://cmsxrootd.crc.nd.edu/" in text and "file:///cms/cephfs/data" in text

def test_manifest_dry_run_is_nonmutating_and_rejects_nonstore(tmp_path):
    manifest = tmp_path / "manifest.tsv"; manifest.write_text("/store/a.root\t12\n/store/b.root\t7\n")
    result = subprocess.run([str(driver), "--dry-run", str(manifest), str(tmp_path / "logs")], text=True, capture_output=True, check=True)
    assert "required_free_bytes=19" in result.stdout.splitlines()[0] and not (tmp_path / "logs").exists()
    bad = tmp_path / "bad.tsv"; bad.write_text("/bad/a.root\t1\n")
    assert subprocess.run([str(driver), "--dry-run", str(bad), str(tmp_path / "badlogs")]).returncode != 0
