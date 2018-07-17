"""Estimate normal vectors of a point set."""


import numpy as np

from .transformation import PCA
from . import (
    assertion,
    Coords,
    distance,
)
from .misc import *

def find_normals(coords, radii, k=5, indices=None, prefered_normal=None):
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
    
    # subset
    if indices is None:
        indices = np.arange(len(coords))
    else:
        indices = assertion.ensure_numvector(indices, max_length=len(coords))
    
    # define prefered normal
    if prefered_normal is None:
        prefered_normal = np.zeros(dim)
        prefered_normal[-1] = 1
    else:
        prefered_normal = assertion.ensure_numvector(length=dim)
    
    # generate normals
    normals = np.zeros((len(indices), dim), dtype=float)
    #ball_gen = coords.indexKD().ball_iter(coords[indices, :], radii)
    ball_gen = coords.indexKD().balls_iter(coords[indices, :], radii)
    #knn_gen = coords.indexKD().knn_iter(
    #        coords[indices, :], k=k, distance_upper_bound=max_distance)
    tic()
    #for i, (dists, nIds) in enumerate(knn_gen):
    for i, nIds in enumerate(ball_gen):
        #nIds = [nIds[i] for i in range(len(nIds)) if dists[i] < max_distance]
        #coords[nIds, :]
        if len(nIds) >= dim:
            normals[i, :] = PCA(coords[nIds, :]).pc(dim)
        if i % 10000 == 0:
            toc()
            tic()
            print(i)
            
                
    # flip normals if required
    dists = distance.snorm(normals - prefered_normal)
    normals[dists > 2] *= -1
    
    return normals
            

def approximate_normals(coords, r, prefered_normal=None):
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
    
    # generate normals
    normals = np.zeros(coords.shape, dtype=float)

    #knn_gen = coords.indexKD().knn_iter(
    #        coords[indices, :], k=k, distance_upper_bound=max_distance)
    tic()
    #for i, (dists, nIds) in enumerate(knn_gen):
    for i in range(len(coords)):
        #nIds = [nIds[i] for i in range(len(nIds)) if dists[i] < max_distance]
        #coords[nIds, :]
        if normals[i].sum() == 0:
            nIds = coords.indexKD().ball(coords[i, :], r)
            if len(nIds) >= dim:
                normals[nIds, :] = PCA(coords[nIds, :]).pc(dim)
        if i % 10000 == 0:
            toc()
            tic()
            print(i)
            
                
    # flip normals if required
    dists = distance.snorm(normals - prefered_normal)
    normals[dists > 2] *= -1
    
    return normals