"""Read geometric panel quality without solving aerodynamic forces."""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = ["nodes", "loops", "surfaces", "nearest_node_distance", "node_distance_q01"]
for surface in range(1, 7):
    FEATURE_NAMES += [f"surface_{surface}_{key}" for key in
                     ("loops", "area_sum", "area_min", "area_q01", "area_median",
                      "edge_min", "shape_max", "nonplanarity_max")]


def mesh_quality(path, area, aspect_ratio):
    lines = Path(path).read_text(encoding="utf8").splitlines()
    if not lines[0].startswith("# vspgeom") or int(lines[1]) != 1:
        raise ValueError("Only single-block vspgeom v3 meshes are supported")
    nodes, loops, surfaces = [int(v) for v in lines[2].split()]
    xyz = np.array([[float(v) for v in row.split()] for row in lines[3:3+nodes]])
    offset = 3+nodes
    if int(lines[offset]) != loops:
        raise ValueError("Inconsistent mesh polygon count")
    connectivity = [[int(v) for v in row.split()] for row in lines[offset+1:offset+1+loops]]
    tags = np.array([int(row.split()[0]) for row in lines[offset+1+loops:offset+1+2*loops]])
    if not set(tags).issubset(set(range(1, 7))):
        raise ValueError("Unexpected geometry surface numbering")
    span = np.sqrt(area*aspect_ratio)
    separation = cKDTree(xyz).query(xyz, k=2)[0][:, 1]/span
    values = [nodes, loops, surfaces, separation.min(), np.quantile(separation, .01)]
    groups = {i: [] for i in range(1, 7)}
    for polygon, tag in zip(connectivity, tags):
        if polygon[0] != len(polygon)-1 or polygon[0] < 3:
            raise ValueError("Malformed geometry polygon")
        points = xyz[np.array(polygon[1:])-1]
        edge = np.roll(points, -1, axis=0)-points
        normal = np.cross(points, np.roll(points, -1, axis=0)).sum(0)/2
        panel_area = np.linalg.norm(normal)
        perimeter = np.linalg.norm(edge, axis=1).sum()
        shape = perimeter**2/(4*np.pi*max(panel_area, 1e-16))
        deviation = (abs((points-points[0]) @ (normal/max(panel_area, 1e-16))).max()/span)
        groups[int(tag)].append((panel_area/area, np.linalg.norm(edge, axis=1).min()/span,
                                 shape, deviation))
    for tag in range(1, 7):
        p = np.asarray(groups[tag], float)
        if not len(p):
            values += [0.]*8
        else:
            values += [len(p), p[:, 0].sum(), p[:, 0].min(), np.quantile(p[:, 0], .01),
                       np.median(p[:, 0]), p[:, 1].min(), p[:, 2].max(), p[:, 3].max()]
    result = np.asarray(values, float)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid geometry quality descriptor")
    return result
