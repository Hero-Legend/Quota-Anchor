from tempprov_audit.cli.build_temporal_snapshot_benchmark import deterministic_fold, pair_id


def test_pair_id_matches_existing_benchmark_convention():
    assert pair_id("C123", "MESH:D456") == "C123__MESH_D456"


def test_deterministic_fold_is_stable_and_bounded():
    value = deterministic_fold("MESH:D008103", 5)
    assert value == deterministic_fold("MESH:D008103", 5)
    assert 0 <= value < 5
