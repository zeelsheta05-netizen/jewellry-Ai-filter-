"""Convert MatrixGold/Rhino .3dm designs into web try-on models.

    .venv-cad/bin/python scripts/cad_to_glb.py FILE.3dm [FILE.3dm ...]
    .venv-cad/bin/python scripts/cad_to_glb.py --category ring --limit 50

Output per design, in data/tryon/models/:
    <slug>.glb   metres, Y-up. Node "metal" plus one node per stone; stones of
                 the same cut share one mesh, so the browser can instance them.
    <slug>.json  category, real size in mm, ring size, stone counts, triangles.

Frame convention (what the try-on page relies on):
    ring      origin = centre of the finger-size circle, +Y = finger axis,
              +Z = towards the head (centre stone), so the head faces the
              back of the hand when +Z is aligned with the hand's normal.
    others    origin = top-centre of the bounding box (hook / bail), +Y up.

Where the geometry comes from (rhino3dm cannot mesh NURBS itself):
    metal   the STL export next to the .3dm when there is one: it is what gets
            cast, so it is complete. Otherwise the .3dm's cached render
            meshes; a file whose metal is only partly meshed is reported as
            "incomplete_mesh" rather than shown with pieces missing.
    stones  the .3dm's stone placements and gem objects (an STL for casting
            has none). A stone saved without a render mesh is rebuilt as the
            convex hull of its facets: cut stones are convex.
Then one piece is kept: the one on the finger-size curve (rings) or the
largest, plus anything within ATTACH_MM of it (a ring head, a drop earring's
links); one earring of a pair; only stones that sit on that metal.

The metal is decimated to MAX_METAL_TRIS: enough to look right on a phone
screen, deliberately too coarse to cast from (IP protection).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import fast_simplification
import numpy as np
import rhino3dm as r
import subprocess
import trimesh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tryon" / "models"
sys.path.insert(0, str(ROOT))
from jewelsearch import storage as ds  # noqa: E402  (the dataset storage: S3 or a folder)


def dataset_name(p: Path) -> str:
    """The dataset name of a local file from fetch() (for reports), else its path."""
    return ds.get().name_of(p) or str(p)
MAX_METAL_TRIS = 60_000
MAX_STONE_TRIS = 2_000
MM = 0.001  # glTF is in metres
GLTFPACK = ROOT / ".tools" / "gltfpack"  # meshoptimizer CLI (MIT), optional

# US ring size -> inner diameter in mm (ISO 8653 table, 0.5 steps)
def us_size_to_mm(size: float) -> float:
    return 11.63 + 0.8128 * size


def mesh_arrays(m: r.Mesh) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([(p.X, p.Y, p.Z) for p in m.Vertices], dtype=np.float64)
    tris = []
    for i in range(m.Faces.Count):
        a, b, c, d = m.Faces[i]
        tris.append((a, b, c))
        if c != d:
            tris.append((a, c, d))
    return v, np.array(tris, dtype=np.int64).reshape(-1, 3)


def geometry_meshes(g) -> list[r.Mesh]:
    if isinstance(g, r.Mesh):
        return [g]
    if isinstance(g, r.Brep):
        out = [f.GetMesh(r.MeshType.Any) for f in g.Faces]
        return [m for m in out if m is not None]
    if isinstance(g, r.Extrusion):
        m = g.GetMesh(r.MeshType.Any)
        return [m] if m is not None else []
    return []


def stone_from_brep(g) -> trimesh.Trimesh | None:
    """A stone saved without a render mesh (common: MatrixGold files whose
    STL export for casting has no stones at all). Cut stones are convex, so
    the convex hull of the facet corners and edges is the stone itself."""
    if isinstance(g, r.Extrusion):
        g = g.ToBrep(True)
    if not isinstance(g, r.Brep):
        return None
    pts = [(v.Location.X, v.Location.Y, v.Location.Z) for v in g.Vertices]
    for e in g.Edges:   # round girdles are curved edges: sample them
        t0, t1 = e.Domain.T0, e.Domain.T1
        pts += [(q.X, q.Y, q.Z) for q in (e.PointAt(t0 + (t1 - t0) * k / 8) for k in range(9))]
    if len(pts) < 4:
        return None
    try:
        hull = trimesh.convex.convex_hull(np.array(pts))
    except Exception:   # flat or degenerate
        return None
    return hull if hull.volume > 0 else None


def to_trimesh_list(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh | None:
    return trimesh.util.concatenate(meshes) if meshes else None


def to_trimesh(meshes: list[r.Mesh]) -> trimesh.Trimesh | None:
    parts = [trimesh.Trimesh(*mesh_arrays(m), process=False) for m in meshes if m.Faces.Count]
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def xform_matrix(x: r.Transform) -> np.ndarray:
    return np.array([[getattr(x, f"M{i}{j}") for j in range(4)] for i in range(4)])


def is_gem_layer(name: str) -> bool:
    return bool(re.search(r"\b(gem|stone|diamond)", name, re.I))


def finger_circle(model, layers):
    """Ring rail curve MatrixGold puts on the 'Finger Sizes' layer."""
    for o in model.Objects:
        g = o.Geometry
        if not isinstance(g, r.Curve) or not g.IsClosed:
            continue
        lname = layers[o.Attributes.LayerIndex]
        label = f"{lname} {o.Attributes.Name or ''}"
        if not re.search(r"finger|ring rail", label, re.I):
            continue
        # sample the curve and fit a circle: centre, radius, plane normal
        t0, t1 = g.Domain.T0, g.Domain.T1
        pts = np.array([[p.X, p.Y, p.Z] for p in (g.PointAt(t0 + (t1 - t0) * k / 64) for k in range(64))])
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        normal = vt[2]
        radius = float(np.linalg.norm(pts - c, axis=1).mean())
        m = re.search(r"(USA?|US)\s*([\d.]+)", label, re.I)
        return {"centre": c, "axis": normal, "radius": radius,
                "us_size": float(m.group(2)) if m else None, "label": label.strip(), "from": "curve"}
    return None


def hole_from_metal(metal: trimesh.Trimesh):
    """Fallback when a ring has no finger-size curve.

    The finger axis is the thinnest principal direction of the metal; the
    hole centre is the point in that cross-section farthest from any metal
    (largest empty circle), and that distance is the inner radius."""
    from scipy.spatial import cKDTree

    pts = metal.vertices
    c0 = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
    axis, u, v = vt[2], vt[0], vt[1]
    flat = np.stack([(pts - c0) @ u, (pts - c0) @ v], axis=1)
    tree = cKDTree(flat)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    best, best_d = None, -1.0
    for step in (40, 20):  # coarse grid, then refine around the best point
        if best is None:
            gx = np.linspace(lo[0], hi[0], step)
            gy = np.linspace(lo[1], hi[1], step)
        else:
            span = (hi - lo) / 20
            gx = np.linspace(best[0] - span[0], best[0] + span[0], step)
            gy = np.linspace(best[1] - span[1], best[1] + span[1], step)
        grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
        d, _ = tree.query(grid)
        i = int(np.argmax(d))
        best, best_d = grid[i], float(d[i])
    centre = c0 + best[0] * u + best[1] * v
    return {"centre": centre, "axis": axis, "radius": best_d, "us_size": None,
            "label": "estimated from metal (no finger-size curve)", "from": "metal"}


def split_stl(path: Path) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh], float]:
    """STL has no layers: stones are the small, convex, flat-faceted shells.

    STL also has no units. Jewellery is 2-80 mm, so a model whose largest
    side is under 4 units is taken to be in inches (or metres below 0.2)."""
    mesh = trimesh.load_mesh(path, process=True)
    side = float(max(mesh.extents))
    scale = 1000.0 if side < 0.2 else 25.4 if side < 4 else 1.0
    shells = mesh.split(only_watertight=False)
    biggest = max(float(max(p.extents)) for p in shells)
    metal, stones = [], []
    for part in shells:
        if len(part.faces) < 4:  # stray triangles from the STL export
            continue
        try:
            hull = part.convex_hull
            convex = hull.volume > 0 and abs(part.volume) / hull.volume > 0.85
        except Exception:  # flat or degenerate shell, cannot be a stone
            convex = False
        faceted = len(part.faces) <= 2000 and len(np.unique(np.round(part.face_normals, 2), axis=0)) <= 200
        small = max(part.extents) < biggest * 0.35
        (stones if convex and faceted and small else metal).append(part)
    return (metal or [mesh]), stones, scale


# Largest side (mm) a real piece of each category can have. Outside this the
# file holds something else (a background plate, a loose part of a set...).
ATTACH_MM = 2.0   # a separate piece this close to the design is part of it
MAX_UNMESHED = 0.05   # share of objects without a render mesh tolerated (a stray construction surface)

SIZE_MM = {"ring": (12, 40), "earring": (4, 100), "pendant": (6, 90),
           "necklace": (60, 450), "bracelet": (45, 120)}
# Measuring for the buy page: a bracelet's CAD is often laid out straight
# (~190 mm), which is the whole piece even though it can't be worn as a try-on.
MEASURE_SIZE_MM = {**SIZE_MM, "bracelet": (45, 230)}


def group_parts(clouds: list[np.ndarray], tol: float) -> list[int]:
    """Union objects whose surfaces come within tol of each other: one group
    per physical piece. Files often hold a pair of earrings, extra variants, a
    loose bail or weight notes in 3D letters next to the design. (Bounding
    boxes were too coarse: curly earrings' boxes overlap without touching.)"""
    from scipy.spatial import cKDTree

    parent = list(range(len(clouds)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(clouds)])
    tree = cKDTree(np.concatenate(clouds))
    for i, pts in enumerate(clouds):
        for hits in tree.query_ball_point(pts, r=tol):
            for j in set(owner[hits]) - {i}:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(clouds))]


def pair_gap(mesh: trimesh.Trimesh):
    """A file holding both earrings of a pair joined into one piece (a shared
    stone or a sprue within the grouping tolerance): find an empty band across
    the middle that splits it into two halves of about the same size.
    Returns (axis, cut value) or None."""
    v = mesh.vertices
    # side by side only: a drop earring stacked vertically must stay whole
    for axis in (0,):
        x = v[:, axis]
        lo, hi = float(x.min()), float(x.max())
        if hi - lo <= 0:
            continue
        counts, edges = np.histogram(x, bins=120, range=(lo, hi))
        best = None
        run = None
        for b in range(30, 90):   # middle half only
            if counts[b] == 0:
                run = (run[0], b) if run else (b, b)
                if best is None or run[1] - run[0] > best[1] - best[0]:
                    best = run
            else:
                run = None
        if best is None:
            continue
        cut = (edges[best[0]] + edges[best[1] + 1]) / 2
        left, right = x[x < cut], x[x >= cut]
        wl, wr = np.ptp(left), np.ptp(right)
        if min(len(left), len(right)) > 0.25 * len(x) and min(wl, wr) > 0.6 * max(wl, wr):
            return axis, cut
    # No empty band (the two touch, or a post crosses the gap): the right half
    # is then a sideways copy of the left one. A single earring is a mirror
    # image of itself at most, never a shifted copy.
    from scipy.spatial import cKDTree
    pts = surface_points(mesh, n=4000)
    x = pts[:, 0]
    cut = (x.min() + x.max()) / 2
    left, right = pts[x < cut], pts[x >= cut]
    if min(len(left), len(right)) < 0.3 * len(pts):
        return None
    # two earrings leave the middle nearly empty (a post at most); a single
    # symmetric one, whose halves also look alike, is solid across its centre
    if np.mean(np.abs(x - cut) < 0.05 * np.ptp(x)) > 0.02:
        return None
    shift = right.mean(axis=0) - left.mean(axis=0)
    shift[1:] = 0   # side by side
    d1 = cKDTree(right).query(left + shift)[0]
    d2 = cKDTree(left).query(right - shift)[0]
    size = float(np.ptp(pts[:, 1]))
    if np.median(d1) < 0.03 * size and np.median(d2) < 0.03 * size:
        return 0, cut
    return None


def surface_points(mesh: trimesh.Trimesh, transform=None, n: int = 400) -> np.ndarray:
    pts = mesh.vertices
    if len(pts) > n:   # vertices plus a surface sample: enough to see contact
        pts = np.concatenate([pts[np.random.default_rng(0).choice(len(pts), n // 2, replace=False)],
                              trimesh.sample.sample_surface(mesh, n // 2, seed=0)[0]])
    return trimesh.transform_points(pts, transform) if transform is not None else pts


def smooth_metal(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices(digits_vertex=4)
    mesh.update_faces(mesh.nondegenerate_faces())
    if len(mesh.faces) > MAX_METAL_TRIS:
        v, f = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
            target_reduction=1 - MAX_METAL_TRIS / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=False)
    # polished metal: smooth normals, keep creases sharper than 35 degrees
    return trimesh.graph.smooth_shade(mesh, angle=math.radians(35))


def facet_stone(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices(digits_vertex=5)
    if len(mesh.faces) > MAX_STONE_TRIS:
        v, f = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
            target_reduction=1 - MAX_STONE_TRIS / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=False)
    mesh.unmerge_vertices()  # flat facets, which is how a cut stone looks
    return mesh


def head_direction(points: np.ndarray, c: np.ndarray, y: np.ndarray):
    """The side of a ring that sticks out furthest from the finger: the head
    (a solitaire's prongs, a cluster, a signet's top). None for a band that
    is the same all the way round."""
    rel = points - c
    perp = rel - np.outer(rel @ y, y)
    r = np.linalg.norm(perp, axis=1)
    far = np.quantile(r, 0.97)
    v = perp[r >= far].mean(axis=0)
    return v if np.linalg.norm(v) > 0.3 * far else None


def frame_for(category: str, metal: trimesh.Trimesh, stones_centre, circle, points=None) -> np.ndarray:
    """4x4 matrix from file coordinates (mm) to the try-on frame (mm)."""
    if category in ("ring", "bracelet") and circle is not None:
        y = circle["axis"] / np.linalg.norm(circle["axis"])
        c = circle["centre"]
        # head direction: where the ring sticks out furthest; for a plain or
        # eternity band, towards the (size-weighted) stones or the heaviest
        # metal. (The plain mean of all stones flipped rings whose band is set
        # with stones all round: the head ended up on the palm side.)
        z = head_direction(points, c, y) if points is not None and len(points) else None
        if z is None:
            target = stones_centre if stones_centre is not None else metal.centroid
            z = target - c
        z = z - y * np.dot(z, y)
        if np.linalg.norm(z) < 1e-6:
            z = np.cross(y, [1, 0, 0]) if abs(y[0]) < 0.9 else np.cross(y, [0, 1, 0])
        z /= np.linalg.norm(z)
        x = np.cross(y, z)
        rot = np.eye(4)
        rot[:3, :3] = np.stack([x, y, z])
        tr = np.eye(4)
        tr[:3, 3] = -c
        return rot @ tr
    lo, hi = metal.bounds
    tr = np.eye(4)
    tr[:3, 3] = -np.array([(lo[0] + hi[0]) / 2, hi[1], (lo[2] + hi[2]) / 2])
    return tr


def shape_signature(mesh: trimesh.Trimesh, unit: float) -> np.ndarray:
    """Spread of the surface along its principal axes, in mm, largest first:
    the same for a piece however it is moved or turned (an STL is often laid
    flat for casting)."""
    pts = trimesh.sample.sample_surface(mesh, 3000, seed=0)[0] * unit
    s = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False) / math.sqrt(len(pts))
    return np.sort(s)[::-1]


def shape_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Relative difference of two shape signatures (0 = same shape)."""
    return float(max(np.max(np.abs(a[:2] - b[:2]) / b[:2]), abs(a[2] - b[2]) / max(b[2], 0.5)))


def stones_sit_on(metal: list[trimesh.Trimesh], stones, placements, cuts, tol: float) -> bool:
    """True when the .3dm's stones sit in the STL's metal (same place, same
    units): most stone centres are within reach of the metal. With no stones
    to check, the files are taken to match."""
    centres = [m.centroid for _, m in stones] + [x[:3, 3] for d, x in placements if d in cuts]
    if not centres:
        return True
    from scipy.spatial import cKDTree
    tree = cKDTree(np.concatenate([surface_points(m) for m in metal]))
    # a stone's centre is up to ~its radius from the metal holding it
    reach = [tol + max(float(max(m.extents)) / 2, 0) for _, m in stones] + \
            [tol + float(np.linalg.norm(x[:3, 0])) * float(max(cuts[d][1].extents)) / 2 for d, x in placements if d in cuts]
    d, _ = tree.query(np.array(centres))
    return float(np.mean(d <= np.array(reach))) >= 0.6


def closed_volume(mesh: trimesh.Trimesh) -> float | None:
    """Enclosed volume of one object (file units cubed), or None when its
    surface is open. Welding isn't needed: a closed surface encloses the same
    volume seen from any origin, an open one doesn't, which is the test."""
    tri = mesh.triangles
    if not len(tri):
        return None

    def signed(o):
        a, b, c = tri[:, 0] - o, tri[:, 1] - o, tri[:, 2] - o
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    lo, hi = mesh.bounds
    v0, v1 = signed(lo), signed(hi + (hi - lo))
    if abs(v0 - v1) > 0.02 * max(abs(v0), 1e-12):
        return None
    return abs(v0)


STONE_SHAPES = [("round", r"round|brill|rnd"), ("oval", r"oval"), ("pear", r"pear"),
                ("marquise", r"marq"), ("emerald", r"emer"), ("princess", r"princ|square(?!.*emer)"),
                ("cushion", r"cush"), ("radiant", r"radi"), ("heart", r"heart"),
                ("baguette", r"bag|taper"), ("asscher", r"assch"), ("trillion", r"tril|trian")]


def shape_of(name: str) -> str | None:
    for shape, pat in STONE_SHAPES:
        if re.search(pat, name, re.I):
            return shape
    return None


def loose_stone(mesh: trimesh.Trimesh, scale: float):
    """Shape, girdle size (mm, long side first), depth and volume of a stone
    that is not a MatrixGold gem block: its thinnest principal axis is the
    depth, and the outline across it tells round from square and long cuts."""
    pts = mesh.vertices * scale
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    ext = np.ptp(proj, axis=0)
    L, W, D = float(ext[0]), float(ext[1]), float(ext[2])
    try:
        hull = mesh.convex_hull
        vol = float(hull.volume) * scale ** 3
        from scipy.spatial import ConvexHull
        fill = ConvexHull(proj[:, :2]).volume / max(L * W, 1e-9)   # 2D hull: .volume is the area
    except Exception:
        return None
    aspect = L / max(W, 1e-9)
    if 0.72 <= fill <= 0.86:
        shape = "round" if aspect < 1.08 else "oval"
    elif fill > 0.86:
        shape = "princess" if aspect < 1.08 else "emerald" if aspect < 2 else "baguette"
    else:
        shape = None   # pear, marquise, heart...: not told apart from the outline
    return shape, (L, W, D), vol


def measure(metal_parts, stone_parts, placements, cuts, scale: float) -> dict:
    """What the buy page needs from the kept piece: metal volume (for the gold
    weight) and every stone's cut, size and volume (for its carats)."""
    vol, area, open_area = 0.0, 0.0, 0.0
    for m in metal_parts:
        a = float(m.area)
        v = closed_volume(m)
        area += a
        if v is None:
            open_area += a
        else:
            vol += v
    groups: dict = {}

    def add(shape, dims, v):
        L, W, D = dims
        key = (shape, round(L, 1), round(W, 1))
        g = groups.setdefault(key, {"shape": shape, "size_mm": [round(L, 2), round(W, 2)],
                                    "depth_mm": round(D, 2), "count": 0, "volume_mm3": 0.0})
        g["count"] += 1
        g["volume_mm3"] += v

    blocks = {}
    for d, x in placements:   # MatrixGold gem blocks: a unit stone, scaled per placement
        name, m = cuts[d]
        if d not in blocks:
            blocks[d] = (float(m.convex_hull.volume), m.extents.copy())
        hv, ext = blocks[d]
        lin = x[:3, :3]
        dims = ext * np.linalg.norm(lin, axis=0) * scale   # depth is the block's Z
        L, W = sorted((float(dims[0]), float(dims[1])), reverse=True)
        add(shape_of(name), (L, W, float(dims[2])), hv * abs(np.linalg.det(lin)) * scale ** 3)
    for m in stone_parts:   # gem objects and STL stone shells
        w = m.copy()
        w.merge_vertices(digits_vertex=4)
        bodies = [b for b in w.split(only_watertight=False) if len(b.faces) >= 4]
        one = [w] if len(bodies) <= 1 else bodies
        for b in one:
            got = loose_stone(b, scale)
            if got is None:   # flat pieces: the stone was saved as loose facets
                got = loose_stone(w, scale)
                one = []
            if got:
                add(*got)
            if not one:
                break
    stones = sorted(groups.values(), key=lambda g: -g["volume_mm3"] / g["count"])
    for g in stones:
        g["volume_mm3"] = round(g["volume_mm3"], 3)
    return {"metal_volume_mm3": round(vol * scale ** 3, 2),
            "metal_open_share": round(open_area / area, 4) if area else 1.0,
            "stone_groups": stones}


def convert(path: Path, category: str, slug: str, stl_only: bool = False, measure_only: bool = False) -> dict:
    """Make the try-on model. measure_only: write nothing and return the kept
    piece's measurements instead (scripts/build_cad_specs.py)."""
    t0 = time.time()
    model = r.File3dm.Read(str(path))
    if model is None:
        return {"status": "unreadable"}
    units = str(model.Settings.ModelUnitSystem).split(".")[-1]
    scale = {"Millimeters": 1.0, "Inches": 25.4, "Centimeters": 10.0, "Meters": 1000.0}.get(units)
    if scale is None:
        return {"status": f"units_{units}"}
    layers = {i: l.FullPath for i, l in enumerate(model.Layers)}

    # objects that are block definitions (stone cuts) are not drawn on their own
    def_member = {str(i) for d in model.InstanceDefinitions for i in d.GetObjectIds()}
    defs = {str(d.Id): d for d in model.InstanceDefinitions}
    by_id = {str(o.Attributes.Id): o for o in model.Objects}

    # MatrixGold puts metal on "Metal .." layers and stones on "Gem ..", but
    # real metal also sits on "Heads", "Layer 05"...: skip only the layers
    # that never hold the piece (cutters, the finger-size rail, notes, lights).
    lname = lambda o: layers[o.Attributes.LayerIndex]
    not_piece = re.compile(r"cutt|finger|dimension|light|note|text|annotation|reference|background|plate|creation|guide|camera", re.I)

    parts, placements, no_mesh = [], [], 0      # parts: (is_stone, trimesh)
    for o in model.Objects:
        g, oid = o.Geometry, str(o.Attributes.Id)
        if oid in def_member:
            continue
        if isinstance(g, r.InstanceReference):
            placements.append((str(g.ParentIdefId), xform_matrix(g.Xform)))
            continue
        gem = is_gem_layer(lname(o))
        if not gem and not_piece.search(lname(o)):
            continue
        meshes = geometry_meshes(g)
        if not meshes:
            stone = stone_from_brep(g) if gem else None
            if stone is not None:
                parts.append((True, stone))
            else:
                no_mesh += isinstance(g, (r.Brep, r.Extrusion))
            continue
        parts.append((gem, to_trimesh(meshes)))

    # one mesh per stone cut, reused by every placement
    cuts: dict[str, tuple[str, trimesh.Trimesh]] = {}
    for def_id, _ in placements:
        if def_id in cuts or def_id not in defs:
            continue
        pieces = []
        for i in defs[def_id].GetObjectIds():
            o = by_id.get(str(i))
            if o is not None:
                pieces += geometry_meshes(o.Geometry)
        m = to_trimesh(pieces)
        if m is None:   # the cut has no render mesh: rebuild it from its facets
            hulls = [stone_from_brep(by_id[str(i)].Geometry) for i in defs[def_id].GetObjectIds() if str(i) in by_id]
            m = to_trimesh_list([h for h in hulls if h is not None])
        if m is not None:
            name = re.sub(r"[^A-Za-z0-9_]+", "_", defs[def_id].Name or "stone").strip("_")
            cuts[def_id] = (name, m)
    placements = [(d, x) for d, x in placements if d in cuts]

    # The STL next to a .3dm is the metal that gets cast, so it is complete;
    # the .3dm's own meshes aren't always (parts saved without a render mesh,
    # metal on odd layers). Use the STL's metal with the .3dm's stones (an STL
    # for casting has none) when both describe the piece in the same place.
    used = path
    stl = next((p for p in path.parent.glob("*.stl")
                if p.stem.lower() == path.stem.lower() and not p.name.startswith("._")), None)
    cad_metal = [m for gem, m in parts if not gem]
    cad_stones = [(True, m) for gem, m in parts if gem]
    stl_shape = None
    if stl is not None:
        metal_shells, stone_shells, stl_scale = split_stl(stl)
        if stl_only:
            parts = [(False, m) for m in metal_shells] + [(True, m) for m in stone_shells]
            placements, cuts, used = [], {}, stl
            if stl_scale != scale:
                model = None
            scale = stl_scale
        elif stl_scale == scale and stones_sit_on(metal_shells, cad_stones, placements, cuts, 1.5 / scale):
            parts = [(False, m) for m in metal_shells] + (cad_stones if cad_stones or placements else
                                                          [(True, m) for m in stone_shells])
            used = stl
        elif not cad_metal or no_mesh > MAX_UNMESHED * len(cad_metal):
            # the .3dm can't be used on its own: the STL with its own stones
            # (an STL moved or laid flat for casting doesn't line up with the
            # .3dm's stones; with the .3dm's metal nearly all meshed, the
            # .3dm alone is the better model, so that case falls through)
            parts = [(False, m) for m in metal_shells] + [(True, m) for m in stone_shells]
            placements, cuts, used = [], {}, stl
            if stl_scale != scale:
                model = None   # its finger-size curve is in other units
            scale = stl_scale
        else:
            # the .3dm on its own; the STL still says which of its pieces is
            # the design that gets made (a .3dm can hold several designs)
            stl_shape = shape_signature(trimesh.util.concatenate(metal_shells), stl_scale)
    elif no_mesh > MAX_UNMESHED * len(cad_metal):
        # part of the metal was saved without a render mesh: never show half a design
        return {"status": "no_render_mesh" if not cad_metal else "incomplete_mesh"}
    if not any(not gem for gem, _ in parts):
        return {"status": "no_metal"}

    # ---- keep one physical piece ----
    clouds = [surface_points(m) for _, m in parts] + [surface_points(cuts[d][1], x, 60) for d, x in placements]
    boxes = [(c.min(axis=0), c.max(axis=0)) for c in clouds]
    group = group_parts(clouds, tol=0.6 / scale)
    n_parts = len(parts)
    groups = {}
    for i, gid in enumerate(group):
        groups.setdefault(gid, []).append(i)
    is_metal = lambda i: i < n_parts and not parts[i][0]
    metal_groups = [ids for ids in groups.values() if any(is_metal(i) for i in ids)]
    if not metal_groups:
        return {"status": "no_metal"}
    # A stone that doesn't quite touch its seat (gap < 1.5 mm) joins that
    # metal piece. Stones farther from every piece (e.g. a set's earrings drawn
    # without metal) are dropped rather than left floating on the necklace.
    # (A bounding-box test failed here: a necklace's box encloses its middle.)
    from scipy.spatial import cKDTree
    metal_pts = [np.concatenate([clouds[i] for i in ids if is_metal(i)]) for ids in metal_groups]
    owner = np.concatenate([np.full(len(p), k) for k, p in enumerate(metal_pts)])
    metal_tree = cKDTree(np.concatenate(metal_pts))
    for ids in [ids for ids in groups.values() if not any(is_metal(i) for i in ids)]:
        dist, idx = metal_tree.query(np.concatenate([clouds[i] for i in ids]))
        if dist.min() < 1.5 / scale:
            k = int(owner[idx[int(np.argmin(dist))]])
            metal_groups[k] = metal_groups[k] + ids

    def extent(ids):
        lo = np.min([boxes[i][0] for i in ids], axis=0)
        hi = np.max([boxes[i][1] for i in ids], axis=0)
        return lo, hi

    circle = None
    if category == "ring" and model is not None:
        circle = finger_circle(model, layers)
    chosen = None
    if circle is not None:   # the ring the finger-size curve belongs to
        for ids in metal_groups:
            lo, hi = extent(ids)
            if np.all(circle["centre"] >= lo - 1) and np.all(circle["centre"] <= hi + 1):
                chosen = ids
                break
    if chosen is None and stl_shape is not None:
        sig = lambda ids: shape_signature(to_trimesh_list([parts[i][1] for i in ids if is_metal(i)]), scale)
        gaps = [(shape_gap(sig(ids), stl_shape), k) for k, ids in enumerate(metal_groups)]
        gap, k = min(gaps)
        if gap > 0.15:   # no piece of the .3dm is the design that is made: use the STL alone
            return convert(path, category, slug, stl_only=True, measure_only=measure_only)
        chosen = metal_groups[k]
    if chosen is None:
        chosen = max(metal_groups, key=lambda ids: np.linalg.norm(np.subtract(*extent(ids)[::-1])))
    # Pieces that nearly touch it are part of it: a ring head or halo, or a
    # drop earring's links, are often modelled a hair apart. A second variant
    # or the other earring of a pair sits well away.
    rest = [ids for ids in metal_groups if ids is not chosen]
    while rest:
        tree = cKDTree(np.concatenate([clouds[i] for i in chosen]))
        near = [ids for ids in rest if tree.query(np.concatenate([clouds[i] for i in ids]))[0].min() < ATTACH_MM / scale]
        if not near:
            break
        chosen = chosen + [i for ids in near for i in ids]
        rest = [ids for ids in rest if not any(ids is n for n in near)]
    dropped = len(rest)
    # every stone must sit on the metal it came with: a chain of stones that
    # only touch each other (a second set, a spare) floats in the air otherwise
    tree = cKDTree(np.concatenate([clouds[i] for i in chosen if is_metal(i)]))
    chosen = [i for i in chosen if is_metal(i) or tree.query(clouds[i])[0].min() < 1.5 / scale]

    metal = to_trimesh_list([parts[i][1] for i in chosen if i < n_parts and not parts[i][0]])
    loose = to_trimesh_list([parts[i][1] for i in chosen if i < n_parts and parts[i][0]])
    placements = [placements[i - n_parts] for i in chosen if i >= n_parts]
    cuts = {d: cuts[d] for d, _ in placements}
    # measured before an earring pair is split: weights are for what is sold
    specs = measure([parts[i][1] for i in chosen if i < n_parts and not parts[i][0]],
                    [parts[i][1] for i in chosen if i < n_parts and parts[i][0]],
                    placements, cuts, scale) if measure_only else None
    pair_in_file = False

    if category == "earring" and metal is not None:
        gap = pair_gap(metal)
        pair_in_file = bool(gap)
        if gap:   # keep one earring of the pair
            axis, cut = gap
            metal = metal.submesh([np.where(metal.triangles_center[:, axis] < cut)[0]], append=True)
            if loose is not None:
                keep = np.where(loose.triangles_center[:, axis] < cut)[0]
                loose = loose.submesh([keep], append=True) if len(keep) else None
            placements = [(d, x) for d, x in placements if x[axis, 3] < cut]
            cuts = {d: cuts[d] for d, _ in placements}
            dropped += 1
            lo_mm, hi_mm = metal.bounds[0] * scale, metal.bounds[1] * scale
        else:
            lo_mm, hi_mm = (v * scale for v in extent(chosen))
    else:
        lo_mm, hi_mm = (v * scale for v in extent(chosen))
    largest = float(np.max(hi_mm - lo_mm))
    lim = (MEASURE_SIZE_MM if measure_only else SIZE_MM).get(category)
    if lim and not lim[0] <= largest <= lim[1]:
        return {"status": "rejected_size", "largest_mm": round(largest, 1)}

    # stones weighted by size, so a centre stone outweighs a band of melee
    stone_pts = [x[:3, 3] for _, x in placements]
    weights = [abs(np.linalg.det(x[:3, :3])) for _, x in placements]
    if loose is not None:
        stone_pts.append(loose.centroid)
        weights.append(float(np.mean(weights)) if weights else 1.0)
    stones_centre = np.average(stone_pts, axis=0, weights=weights) if stone_pts else None
    if category in ("ring", "bracelet") and circle is None:
        circle = hole_from_metal(metal)
    shape_pts = [metal.vertices] + [trimesh.transform_points(cuts[d][1].vertices, x) for d, x in placements]
    if loose is not None:
        shape_pts.append(loose.vertices)
    frame = frame_for(category, metal, stones_centre, circle, np.concatenate(shape_pts))
    to_gltf = np.diag([scale * MM] * 3 + [1.0]) @ frame  # mm frame -> metres

    if measure_only:
        # size of one piece (one earring) in the try-on frame, stones included
        dims = np.ptp(trimesh.transform_points(np.concatenate(shape_pts), frame), axis=0) * scale
        out = {"status": "ok", "slug": slug, "category": category,
               "source": dataset_name(used),
               "size_mm": [round(float(v), 2) for v in dims], **specs}
        if category == "earring":
            out["pair_in_file"] = pair_in_file
        if circle is not None:
            inner = circle["radius"] * 2 * scale
            out["ring"] = {"inner_diameter_mm": round(inner, 2), "estimated": circle["from"] == "metal",
                           "us_size": circle["us_size"] or round((inner - 11.63) / 0.8128 * 2) / 2}
        return out

    scene = trimesh.Scene()
    metal = smooth_metal(metal)
    metal.apply_transform(to_gltf)
    metal.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        name="metal", baseColorFactor=[0.96, 0.78, 0.45, 1.0], metallicFactor=1.0, roughnessFactor=0.12))
    scene.add_geometry(metal, geom_name="metal", node_name="metal")

    stone_mat = trimesh.visual.material.PBRMaterial(
        name="stone", baseColorFactor=[1, 1, 1, 1], metallicFactor=0.0, roughnessFactor=0.0)
    stone_tris = 0
    for def_id, (name, m) in cuts.items():
        m = facet_stone(m)
        m.visual = trimesh.visual.TextureVisuals(material=stone_mat)
        cuts[def_id] = (name, m)
        stone_tris += len(m.faces)
    # Block geometry is in the file's units (MatrixGold uses a 10 mm unit
    # stone scaled per placement). Store it in metres so every mesh shares a
    # few-cm quantisation grid; a +-5 m stone would leave the metal ~0.6 mm steps.
    unscale = np.diag([1 / MM] * 3 + [1.0])
    added = set()
    for def_id, (name, m) in cuts.items():
        m.apply_scale(MM)
    for k, (def_id, x) in enumerate(placements):
        name, m = cuts[def_id]
        geom = f"stone_{name}"
        if geom not in added:   # one mesh per cut; every placement is a node pointing at it
            scene.geometry[geom] = m
            added.add(geom)
        scene.graph.update(frame_to=f"{geom}_{k}", frame_from=scene.graph.base_frame,
                           matrix=to_gltf @ x @ unscale, geometry=geom)
    if loose is not None:
        m = facet_stone(loose)
        m.apply_transform(to_gltf)
        m.visual = trimesh.visual.TextureVisuals(material=stone_mat)
        scene.add_geometry(m, geom_name="stone_loose", node_name="stone_loose")

    OUT.mkdir(parents=True, exist_ok=True)
    glb = OUT / f"{slug}.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    if GLTFPACK.exists():
        # meshopt compression + quantisation; repeated stones become
        # EXT_mesh_gpu_instancing (three.js turns them into an InstancedMesh)
        raw = glb.with_suffix(".raw.glb")
        glb.rename(raw)
        subprocess.run([str(GLTFPACK), "-i", str(raw), "-o", str(glb), "-cc", "-mi", "-km"],
                       check=True, capture_output=True)
        raw.unlink()

    ext = (metal.bounds[1] - metal.bounds[0]) / MM
    meta = {
        "status": "ok",
        "slug": slug,
        "source": dataset_name(used),
        "category": category,
        "size_mm": [round(float(v), 2) for v in ext],
        "stones": {name: sum(1 for d, _ in placements if cuts[d][0] == name) for _, (name, _) in cuts.items()},
        "loose_stone_tris": 0 if loose is None else int(len(loose.faces)),
        "other_pieces_dropped": dropped,
        "metal_tris": int(len(metal.faces)),
        "stone_tris_unique": int(stone_tris),
        "glb_kb": round(glb.stat().st_size / 1024),
        "seconds": round(time.time() - t0, 1),
    }
    if circle is not None:
        inner = circle["radius"] * 2 * scale
        meta["ring"] = {
            "inner_diameter_mm": round(inner, 2),
            "estimated": circle["from"] == "metal",
            "us_size": circle["us_size"] or round((inner - 11.63) / 0.8128 * 2) / 2,
            "label": circle["label"],
            # outer radius of the shank at the bottom (-Z), used for the finger occluder check
            "band_width_mm": round(float(ext[1]), 2),
        }
    (OUT / f"{slug}.json").write_text(json.dumps(meta, indent=1))
    return meta


CATEGORY_WORDS = {
    "ring": r"ring(?!s? *not)|/LR-|DDLR|SBMR|GGR",
    "earring": r"earring",
    "pendant": r"pand|pend",
    "necklace": r"neckl",
    "bracelet": r"brel|brace",
}


def guess_category(p: Path) -> str:
    s = str(p)
    if re.search(r"earring", s, re.I):
        return "earring"
    for cat, pat in CATEGORY_WORDS.items():
        if re.search(pat, s, re.I):
            return cat
    return "unknown"


def slug_for(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem).strip("_")


def fetch(rel: str) -> Path:
    """A dataset file as a local file (from the storage cache when the dataset is
    in S3). For a .3dm, the same-named .stl in its folder comes along, because
    convert() uses that STL as the metal when it is there."""
    st = ds.get()
    local = st.local_copy(rel)
    if rel.lower().endswith(".3dm") and not isinstance(st, ds.LocalStorage):
        folder, _, name = rel.rpartition("/")
        stem = name[:-4].lower()
        for other in st.walk(folder, exts={".stl"}):
            o_dir, _, o_name = other.rpartition("/")
            if o_dir == folder and o_name[:-4].lower() == stem:
                stl = st.local_copy(other)
                if stl.parent != local.parent:   # mid-move: one file in the bucket, the other still on the SSD
                    shutil.copy2(stl, local.parent / stl.name)
    return local


def find_files(category: str, limit: int) -> list[Path]:
    found = []
    for rel in ds.get().walk(exts={".3dm"}):
        p = Path(rel)
        if "photo" in p.stem.lower() or p.stem.endswith("-r"):
            continue
        if category in ("all", guess_category(p)):
            found.append(fetch(rel))
            if len(found) >= limit:
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--category", default=None, help="ring|earring|pendant|necklace|bracelet|all")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    files = args.files or find_files(args.category or "all", args.limit)
    report = OUT.parent / "convert_report.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    with report.open("a") as rep:
        for p in files:
            cat = args.category if args.category not in (None, "all") else guess_category(p)
            try:
                meta = convert(p, cat, slug_for(p))
            except Exception as e:  # keep the batch going, log the failure
                meta = {"status": f"error: {e!r}"}
            meta.setdefault("source", str(p))
            rep.write(json.dumps(meta) + "\n")
            print(f"{meta['status']:<22} {cat:<9} {meta.get('glb_kb', '-'):>6} KB "
                  f"{meta.get('metal_tris', '-'):>7} tris  {meta.get('seconds', '-')}s  {p.name}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
