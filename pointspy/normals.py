"""Estimate normal vectors of a point set."""


import numpy as np

from . import (
    assertion,
    Coords,
    transformation,
    distance,
)

def find_normals(coords, radii, prefered_normal=None):
    """Calculate normals from coordinates based on neighbouring points. 
    
    Parameters
    ----------
    coords : array_like(Number, shape=(n, k))
        Represents `n` points of `k` dimensions.
    radii : positive Number or array_like(Number, shape=(n))
        Radius or radii to select neighbouring points.
        
    Returns
    -------
    array_like(Number, shape=(n, k))
        Desired normals of coordinates `coords`.
    
    Examples
    --------
    
    Two dimensional normals.
    
    >>> coords = [(0, 0), (1, 1), (2, 3), (3, 3), (4, 2), (5, 1), (5, 0)]
    >>> normals = find_normals(coords, 1.5)
    >>> print(np.round(normals, 2))
    [[-0.71  0.71]
     [-0.71  0.71]
     [ 0.    1.  ]
     [ 0.47  0.88]
     [ 0.71  0.71]
     [ 0.88  0.47]
     [ 1.    0.  ]]
    
    """
    coords = Coords(coords)
    dim = coords.dim
    
    # define prefered normal
    if prefered_normal is None:
        prefered_normal = np.zeros(dim)
        prefered_normal[-1] = 1
    else:
        prefered_normal = assertion.ensure_numvector(length=dim)
    
    normals = np.zeros(coords.shape, dtype=float)

    # generate normals
    ball_gen = coords.indexKD().ball_iter(coords, radii)
    for i, nIds in enumerate(ball_gen):

        if len(nIds) >= dim:

            # normal as last principal component of PCA
            pca = transformation.PCA(coords[nIds, :])
            normal = pca.pc(dim)
            
            # flip normal if required
            dist = distance.snorm(prefered_normal - normal)
            if dist > 2:
                normal = -normal
            normals[i, :] = normal
        
    return normals
            