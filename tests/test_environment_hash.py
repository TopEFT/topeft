from pathlib import Path
import hashlib

EXPECTED_SHA256 = "e1bfdc6116d857c6256f718b5f01433e6525e24bedd441a62e3793b644accf22"


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
