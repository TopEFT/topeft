from pathlib import Path
import hashlib

EXPECTED_SHA256 = "f48088dc786c70552262b198c56c39830b4ac1af30583eee1b559f5804ff9020"


def test_environment_spec_matches_ttbareft():
    """Ensure the local Conda spec mirrors ttbarEFT's coffea2025 baseline."""

    env_path = Path(__file__).resolve().parents[1] / "environment.yml"
    digest = hashlib.sha256(env_path.read_bytes()).hexdigest()

    assert (
        digest == EXPECTED_SHA256
    ), (
        "environment.yml drifted from the ttbarEFT coffea2025 specification. "
        "Update the file and refresh EXPECTED_SHA256 to match upstream. "
        f"Observed {digest}."
    )
