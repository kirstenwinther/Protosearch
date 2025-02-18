import numpy as np
import scipy
from shapely.geometry import Polygon
from catkit.gen.utils.connectivity import get_cutoff_neighbors, get_voronoi_neighbors

def expand_cell(atoms, cutoff=None, padding=None):
    """
    Copy from Catkit connectivity utils (written by Jacob Boes)
    Return Cartesian coordinates atoms within a supercell
    which contains repetitions of the unit cell which contains
    at least one neighboring atom.

    Parameters
    ----------
    atoms : Atoms object
        Atoms with the periodic boundary conditions and unit cell
        information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.
    padding : ndarray (3,)
        Padding of repetition of the unit cell in the x, y, z
        directions. e.g. [1, 0, 1].

    Returns
    -------
    index : ndarray (N,)
        Indices associated with the original unit cell positions.
    coords : ndarray (N, 3)
        Cartesian coordinates associated with positions in the
        supercell.
    offsets : ndarray (M, 3)
        Integer offsets of each unit cell.
    """
    cell = atoms.cell
    pbc = atoms.pbc
    pos = atoms.positions

    if padding is None and cutoff is None:
        diags = np.sqrt((
            np.dot([[1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [-1, -1, 1]],
                   cell)**2).sum(1))

        if pos.shape[0] == 1:
            cutoff = max(diags) / 2.
        else:
            dpos = (pos - pos[:, None]).reshape(-1, 3)
            Dr = np.dot(dpos, np.linalg.inv(cell))
            D = np.dot(Dr - np.round(Dr) * pbc, cell)
            D_len = np.sqrt((D**2).sum(1))

            cutoff = min(max(D_len), max(diags) / 2.)

    latt_len = np.sqrt((cell**2).sum(1))
    V = abs(np.linalg.det(cell))
    padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                     (V * latt_len)), dtype=int)

    offsets = np.mgrid[
        -padding[0]:padding[0] + 1,
        -padding[1]:padding[1] + 1,
        -padding[2]:padding[2] + 1].T
    tvecs = np.dot(offsets, cell)
    coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]

    ncell = np.prod(offsets.shape[:-1])
    index = np.arange(len(atoms))[None, :].repeat(ncell, axis=0).flatten()
    coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
    offsets = offsets.reshape(ncell, 3)

    return index, coords, offsets


def get_area_neighbors(atoms, cell_cutoff=15, cutoff=None,
                       return_std_con=False):
    """
    Extention of the CatKit Voronoi connectivity.

    Return the connectivity matrix from the Voronoi
    method weighted by the area of the Voronoi vertices,
    which changes smoothly with small geometry changes.
    Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cell_cutoff : float
        Cutoff for supercell repetition used for Voronoi
    cutoff : float
        Maximum atomic bond distance to consider for connectivity

    Returns
    -------
    connectivity : ndarray (n, n) of floats
       Weighted number of edges formed between atoms in a system.
    connectivity_int : ndarray (n, n) of ints
       Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms,
                                         cutoff=cell_cutoff)
    L = int(len(offsets) / 2)

    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    areas = []
    area_indices = []
    connectivity = np.zeros((len(atoms), len(atoms)))
    connectivity_int = np.zeros((len(atoms), len(atoms)))
    try:
        voronoi = scipy.spatial.Voronoi(coords, qhull_options='QbB Qc Qs')
        points = voronoi.ridge_points
    except:
        print('Voronoi failed')
        if not return_std_con:
            return connectivity
        else:
            return connectivity, connectivity_int.astype(int)

    Voro_cell = voronoi.points.ptp(axis=0)
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                ("scipy.spatial.Voronoi returned an atom which has "
                 "no neighbors. This may result in incorrect connectivity."))
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            if cutoff:
                indices = indices[np.where(dist[indices] < cutoff)[0]]
            d = dist[indices]
            count = len(d)
            u, v = edge
            area_indices += [sorted([u, v])]
            edge_areas = []
            for ii, ind in enumerate(ridge_indices[indices]):
                vertices = np.array([list(voronoi.vertices[v]) for v in
                                     voronoi.ridge_vertices[ind]])
                if len(vertices) < 3:
                    continue

                vertices *= Voro_cell

                area = get_weighted_area(vertices, d[ii])
                if area is None:
                    continue

                connectivity[u][v] += area
                connectivity[v][u] += area

            connectivity_int[u][v] += count
            connectivity_int[v][u] += count

    connectivity /= 2

    if not return_std_con:
        return connectivity
    else:
        connectivity_int /= 2
        return connectivity, connectivity_int.astype(int)


def get_voronoi_neighbors(atoms, cutoff=10, return_distances=False):
    """
    Based on connectivity from CatKit, with a fix for the distance
    cutoff.
    Return the connectivity matrix from the Voronoi
    method. Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.

    Returns
    -------
    connectivity : ndarray (n, n)
        Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms, cutoff=cutoff)

    xm, ym, zm = np.max(coords, axis=0) - np.min(coords, axis=0)

    L = int(len(offsets) / 2)
    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    voronoi = scipy.spatial.Voronoi(coords, qhull_options='QbB Qc Qs')
    points = voronoi.ridge_points

    connectivity = np.zeros((len(atoms), len(atoms)))
    distances = []
    distance_indices = []
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                ("scipy.spatial.Voronoi returned an atom which has "
                 "no neighbors. This may result in incorrect connectivity."))
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            d = dist[indices][np.where(dist[indices] < cutoff)[0]]
            count = len(d)
            if count == 0:
                continue

            u, v = edge

            distance_indices += [sorted([u, v])]
            distances += [sorted(d)]

            connectivity[u][v] += count
            connectivity[v][u] += count

    connectivity /= 2
    if not return_distances:
        return connectivity.astype(int)

    if len(distances) > 0:
        distance_indices, unique_idx_idx = \
            np.unique(distance_indices, axis=0, return_index=True)
        distance_indices = distance_indices.tolist()

        distances = [distances[i] for i in unique_idx_idx]

    pair_distances = {'indices': distance_indices,
                      'distances': distances}

    return connectivity.astype(int), pair_distances


def get_weighted_area(vertices, d):
    """ helper method for voronoi area"""
    center = vertices[0, :]
    vertices -= center

    # Plane of polygon in 3d defined by unit vectors ux and uy
    ux = vertices[1, :].copy()
    ux /= np.linalg.norm(ux)
    vy = vertices[2, :].copy()

    # ensure uy is orthogonal to ux
    vyproj = np.dot(vy, ux) * ux
    uy = vy - vyproj
    uy /= np.linalg.norm(uy)

    # Project all vectors onto plane to get 2D coordinates
    Xs = np.dot(vertices, ux)
    Ys = np.dot(vertices, uy)

    poly_coords = [[Xs[i], Ys[i]] for i in range(len(Xs))]
    polygon = Polygon(poly_coords)

    try:
        area = polygon.area
    except:
        return None

    perimeter = polygon.length

    # Scale area with bond distance and perimeter
    weight = d * perimeter / 4
    area /= weight

    return area


def get_connections(atoms, decimals=1):

    #connectivity = get_voronoi_neighbors(atoms) #
    connectivity = get_cutoff_neighbors(atoms, scale_cov_radii=10)

    # get_area_neighbors(atoms)  # get_cutoff_neighbors(atoms)

    atoms_connections = {}

    for i in range(len(atoms)):
        for j in range(len(atoms)):
            symbols = '-'.join(sorted([atoms[i].symbol, atoms[j].symbol]))
            if not symbols in atoms_connections:
                atoms_connections[symbols] = [0 for n in range(len(atoms))]

            if not atoms[i].symbol == atoms[j].symbol:
                idx = np.argsort([atoms[i].symbol, atoms[j].symbol])
                k = [i, j][idx[-1]]
            else:
                k = i

            if not connectivity[i][j]:
                continue
            atoms_connections[symbols][k] += \
                connectivity[i][j] / 2

    final_connections = {}
    for key, value in atoms_connections.items():
        value = [v for v in value if v > 0]
        if len(value) > 0:
            m = np.mean(value)
            std = np.std(value)
        else:
            m = 0
            std = 0
        final_connections[key + '_mean'] = np.round(m, decimals)
        final_connections[key + '_std'] = np.round(std, decimals)
    return final_connections
