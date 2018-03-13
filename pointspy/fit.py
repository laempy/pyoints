import numpy as np
from scipy import optimize

from . import nptools
from . import transformation
from . import _assertion



def ball(coords, weights=1.0):
    """Least square fitting of a sphere to a set of points.
    
    Parameters
    ----------
    coords : (n,k), array_like
        Represents n data points of `k` dimensions.
    weights : (k+1,k+1), array_like
        Transformation matrix.
    
    Returns
    -------
    center : (k), np.ndarray
        Center of the sphere.
    R : float
        Radius of the sphere.
        
    Examples
    --------
    
    Draw points on a half circle with radius 5 and cener (2,4) and try to 
    dertermine the circle parameters.
    
    >>> x = np.arange(-1,1,0.1)
    >>> y = np.sqrt(5**2 - x**2)
    >>> coords = np.array([x,y]).T + [2,4]
    >>> center,R = ball(coords)
    >>> print center
    [2. 4.]
    >>> print np.round(R,2)
    5.0
    
    """
    
    coords = _assertion.ensure_coords(coords)
    
    assert ( isinstance(weights,int) or isinstance(weights,float) ) or \
        ( hasattr(weights,'__len__') and len(weights) == coords.shape[0] )


    # TODO radius und mittelpunkt vorgeben

    # http://www.arndt-bruenner.de/mathe/scripts/kreis3p.htm
    dim = coords.shape[1]

    # mean-centering to avoid overflow errors
    c = coords.mean(0)  
    cCoords = coords - c

    A = transformation.homogenious(cCoords,value=1)
    B = (cCoords**2).sum(1)

    A = (A.T * weights).T
    B = B * weights

    p = np.linalg.lstsq(A, B, rcond=-1)[0]

    bCenter = 0.5 * p[:dim]
    R = np.sqrt((bCenter**2).sum() + p[dim])

    center = bCenter + c
    return center, R



def cylinder(origin, coords, p, th):
    """Fit a cylinder to points.
    
    Parameters
    ----------
    TODO
    TODO referenz
    
    Examples
    --------
    >>> r = 2
    >>> from pointspy.polar import
    >>> x = np.arange(-1,1,0.01)
    >>> y = np.sqrt(1**2 - x**2)
    >>> y[::2] = - y[::2]
    >>> x = x * r
    >>> y = y * r
    >>> z = np.ones(len(x)) * 5
    >>> z[::2] = -5
    >>> coords = np.array([x,y,z]).T
    >>> T = transformation.matrix(t=[1,2,3],r=[0.15,0.2,0.0])
    >>> print np.round(T,2)
    T
    >>> rCoords = transformation.transform(coords,T)
    >>> M, center, r, success = cylinder(None,rCoords,[0,0,0,0,0],None)
    >>> print success
    True
    >>> print np.round(r,2)
    3.0
    >>> print np.round(center,2)
    [1. 2. 3.]
    >>> print np.round(M,2)
    [[ 1.  0. -0. -1.]
     [ 0.  1.  0. -2.]
     [ 0. -0.  1. -3.]
     [ 0.  0.  0.  1.]]
    >>> print np.round(transformation.transform(rCoords,M,inverse=True),2)

    
    # https://stackoverflow.com/questions/42157604/how-to-fit-a-cylindrical-model-to-scattered-3d-xyz-point-data/42163007
    # https://stackoverflow.com/questions/43784618/fit-a-cylinder-to-scattered-3d-xyz-point-data-with-python
    # https://de.mathworks.com/help/vision/ref/pcfitcylinder.html?requestedDomain=www.mathworks.com
    # Chan_2012a !!!

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
    coords = _assertion.ensure_coords(coords)
    assert coords.shape[1] == 3
    assert coords.shape[0] >= 3
    
    
    c = coords.mean(0)
    xyz = coords - c

    


    def fitfunc(p, x, y, z):
        # fit function
        return (
                    - np.cos(p[3]) * (p[0] - x)
                    - z * np.cos(p[2]) * np.sin(p[3]) 
                    - np.sin(p[2]) * np.sin(p[3]) * (p[1] - y)
                )**2 + (
                    z * np.sin(p[2]) 
                    - np.cos(p[2]) * (p[1] - y)
                )**2
                
    def errfunc(p, x, y, z):
        return fitfunc(p, x, y, z) - p[4]**2  # error function
    

    x,y,z = nptools.colzip(xyz)

    m = np.pi * 0.5
    
    # test three perpendicular directions
    ps = [(0,0,0,0,0),(0,0,0,0,m),(0,0,0,m,0)]
    for p in ps:
        est_p, success = optimize.leastsq(errfunc, p, args=(x, y, z),maxfev=100000000)
        print success
        if success in (1,2,3,4):
            break
    #print errfunc(las.coords)

    r = est_p[4]

    T0 = transformation.t_matrix(c)
    R = transformation.r_matrix([est_p[2], est_p[3], 0])
    T1 = transformation.t_matrix([est_p[0], est_p[1], 0])
    #T0 = transformation.tMatrix(-origin)
    #M = R*T1
    M = R * T0 * T1
    M = T0 * T1 * R
    #M = R * T1


    M = transformation.LocalSystem(M)
    center = M.to_global([0, 0, 0])
    #print transformation.decomposition(M)

    print (coords[:,2]<0).sum()
    print (coords[:,2]>0).sum()

    return M, center, r, success == 0



class PCA(transformation.LocalSystem):

    def __init__(self,coords):
        pass

    def __new__(cls, coords):
    
        coords = _assertion.ensure_coords(coords)
        
        
        center = coords.mean(0)
        dim = len(center)

        cCoords = coords - center
        if cCoords.shape[0] == 2:
            # Add dimension if neccessary
            cCoords = transformation.homogenious(cCoords,value=0)

        covM = np.cov(cCoords, rowvar=False)
        evals, evecs = np.linalg.eigh(covM)
        idx = np.argsort(evals)[::-1]
        pComponents = evecs[:, idx].T

        # Orientation
        pc1 = pComponents[0, :]
        mIndex = np.argmax(np.abs(pc1))
        if pc1[mIndex] < 0:
            pComponents = -pComponents

        # Transformation matrix
        T = np.matrix(np.identity(dim + 1))
        T[0:dim, 0:dim] = pComponents
        T = T * transformation.tMatrix(-center)

        return transformation.LocalSystem(T).view(cls)



