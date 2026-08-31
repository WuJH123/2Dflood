from hydrograph.data import UrbanFloodBenchRepository
from hydrograph.schema import validate_split_schema


def test_schema_and_loader(synthetic_root):
    repo = UrbanFloodBenchRepository(synthetic_root)
    models = repo.model_dirs()
    assert len(models) == 3
    for m in models:
        report = validate_split_schema(m / "train")
        assert report.ok, (report.missing_files, report.missing_columns)
        g = repo.load_static(m.name)
        assert g.n1 == 3 and g.n2 == 8
        assert g.node2_static.shape[1] == 9
        assert g.edge1_static.shape[1] == 6
        assert g.edge2_static.shape[1] == 5
    refs = repo.list_events("train")
    e = repo.load_event(refs[0])
    assert e.node1.shape == (20, 3, 2)
    assert e.node2.shape == (20, 8, 3)
    assert e.edge1.shape[-1] == 2 and e.edge2.shape[-1] == 2
