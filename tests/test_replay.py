from pathlib import Path

from scramjet_rl.cfd.replay import rollout_to_manifest


def test_rollout_to_manifest_preserves_ramp_angle(tmp_path: Path):
    rollout = tmp_path / "rollout.csv"
    output = tmp_path / "manifest.csv"
    rollout.write_text(
        "episode,step,mach,altitude_m,ramp_angle_deg\n0,1,6.1,21000,11.5\n",
        encoding="utf-8",
    )

    rollout_to_manifest(rollout, output)

    text = output.read_text(encoding="utf-8")
    assert "replay_00000" in text
    assert "11.5" in text
