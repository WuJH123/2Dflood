import torch

from hydrograph.data import StaticGraph
from hydrograph.physics import surface_mass_residual


def test_exact_rainfall_mass_balance():
    g = StaticGraph(
        "X", torch.zeros(1, 4), torch.tensor([[0., 0., 120., .04, 0., 0., 0., 0., 0.]]),
        torch.tensor([0]), torch.tensor([0]), torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(2, 0, dtype=torch.long), torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, 6), torch.zeros(0, 5),
        ["depth", "invert_elevation", "surface_elevation", "base_area"],
        ["position_x", "position_y", "area", "roughness", "min_elevation", "elevation", "aspect", "curvature", "flow_accumulation"],
        ["relative_position_x", "relative_position_y", "length", "diameter", "roughness", "slope"],
        ["relative_position_x", "relative_position_y", "face_length", "length", "slope"],
        torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
    )
    current2 = torch.tensor([[0., 0., 0.]])
    next2 = torch.tensor([[1., 0., 10.]])  # 1 inch * 120 ft2 = 10 ft3
    next1 = torch.tensor([[0., 0.]])
    e2 = torch.zeros(0, 2)
    r = surface_mass_residual(g, current2, next1, next2, e2, 300.)
    assert r.local_relative.item() < 1e-7
    assert r.global_relative.item() < 1e-7
