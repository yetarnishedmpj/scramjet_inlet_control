from pathlib import Path

from scramjet_rl.cfd.cases import materialize_cases
from scramjet_rl.cfd.manifest import write_sweep_manifest


def test_materialize_cases_replaces_placeholders(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    template = tmp_path / "template"
    output = tmp_path / "cases"
    template.mkdir()
    (template / "controlDict").write_text("mach={{MACH}}\nangle={{RAMP_ANGLE_DEG}}\n", encoding="utf-8")
    write_sweep_manifest(manifest, count=1, seed=3)

    cases = materialize_cases(manifest, template, output)

    assert len(cases) == 1
    rendered = (cases[0] / "controlDict").read_text(encoding="utf-8")
    assert "{{MACH}}" not in rendered
    assert "{{RAMP_ANGLE_DEG}}" not in rendered
