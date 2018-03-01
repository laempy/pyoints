import numpy as np
from scipy.optimize import leastsq

from . import transformation



def ball(coords, weights=1):

    # TODO radius und mittelpunkt vorgeben

    # http://www.arndt-bruenner.de/mathe/scripts/kreis3p.htm
    N, dim = coords.shape

    c = coords.mean(0)  # mean-centering to avoid overflow
    cCoords = coords - c

    A = np.vstack([-cCoords.T, np.ones(N)]).T
    B = -(cCoords**2).sum(1)

    A = (A.T * weights).T
    B = B * weights

    p = np.linalg.lstsq(A, B)[0]

    bCenter = 0.5 * p[0:dim]
    R = np.sqrt((bCenter**2).sum() - p[dim])

    center = bCenter + c
    return center, R


def PCA(coords):
    center = coords.mean(0)
    dim = len(center)

    # Add values if neccessary
    cCoords = coords - center
    if cCoords.shape[0] == 2:
        cCoords = np.vstack((cCoords, np.zeros(dim)))

    # PCA for multidimensional regression
    #from sklearn import decomposition
    # pca=decomposition.PCA(n_components=dim,copy=False)
    # pca.fit(cCoords)
    # pComponents=pca.components_
    #print pComponents
    #print pca.explained_variance_
    #print pca.explained_variance_ratio_

    covM = np.cov(cCoords, rowvar=False)
    evals, evecs = np.linalg.eigh(covM)
    idx = np.argsort(evals)[::-1]
    pComponents = evecs[:, idx].T
    #print pComponents

    # Orientation
    pc1 = pComponents[0, :]
    mIndex = np.argmax(np.abs(pc1))
    if pc1[mIndex] < 0:
        pComponents = -pComponents

    # Transformation matrix
    T = np.matrix(np.identity(dim + 1))
    T[0:dim, 0:dim] = pComponents
    T = T * transformation.tMatrix(-center)

    return transformation.LocalSystem(T)


def cylinder(origin, coords, p, th):
    # https://stackoverflow.com/questions/42157604/how-to-fit-a-cylindrical-model-to-scattered-3d-xyz-point-data/42163007
    # https://stackoverflow.com/questions/43784618/fit-a-cylinder-to-scattered-3d-xyz-point-data-with-python
    # https://de.mathworks.com/help/vision/ref/pcfitcylinder.html?requestedDomain=www.mathworks.com
    # Chan_2012a !!!
    """
    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder

    th, threshold for the convergence of the least squares

    """
    c = coords.mean(0)
    xyz = coords - c

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    def fitfunc(p, x, y, z): return (- np.cos(p[3]) * (p[0] - x) - z * np.cos(p[2]) * np.sin(p[3]) - np.sin(
        p[2]) * np.sin(p[3]) * (p[1] - y))**2 + (z * np.sin(p[2]) - np.cos(p[2]) * (p[1] - y))**2  # fit function

    def errfunc(p, x, y, z): return fitfunc(
        p, x, y, z) - p[4]**2  # error function

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

    #print errfunc(las.coords)
    #print 'success %s'%success

    r = est_p[4]

    T0 = transformation.tMatrix(-c)
    R = transformation.rMatrix([-est_p[2], -est_p[3], 0])
    T1 = transformation.tMatrix([-est_p[0], -est_p[1], 0])
    #T0 = transformation.tMatrix(-origin)
    #M = R*T1
    M = R * T0 * T1

    M = transformation.LocalSystem(M)
    center = M.toGlobal([0, 0, 0])

    return M, center, r, success

    # return M,r
