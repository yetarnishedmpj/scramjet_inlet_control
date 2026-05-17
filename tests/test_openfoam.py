from scramjet_rl.cfd.openfoam import check_openfoam


def test_check_openfoam_reports_missing_solver():
    result = check_openfoam("definitely_not_a_real_openfoam_solver")
    assert not result.ok
    assert "not found" in result.message
