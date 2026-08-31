import pytest

from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.split import make_manifest


def test_loco_no_target_leakage(synthetic_root):
    repo = UrbanFloodBenchRepository(synthetic_root)
    refs = repo.list_events("train")
    m = make_manifest(refs, "leave_one_city_out", 42, target_model="Model_3")
    m.validate_no_leakage()
    assert all(r.model_id != "Model_3" for r in m.train)
    assert all(r.model_id == "Model_3" for r in m.test)


def test_split_detects_overlap(synthetic_root):
    repo = UrbanFloodBenchRepository(synthetic_root)
    refs = repo.list_events("train")
    m = make_manifest(refs, "event_holdout", 1)
    m.val.append(m.train[0])
    with pytest.raises(ValueError):
        m.validate_no_leakage()
