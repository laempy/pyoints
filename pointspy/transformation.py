import numpy as np

from . import distance


# TODO documentation


def transform(coords, T, inverse=False):
    """Applies a linear transformation to coordinates
    
    Parameters
    ----------
    coords: (n,k), `array_like`
        Represents n data points of k dimensions.
    T: (k+1,k+1), `array_like`
        Transformation matrix.
    
    Returns
    -------
    coords: (n,k), `array_like`
        Transformed coordinates. If 
    """

    if inverse:
        T = np.linalg.inv(T)
    T = np.asarray(T)
    assert T.shape > 0
    assert T.shape[0] == T.shape[1]

    H = homogenious(coords)
    HT = np.dot(H, T.T)
    if len(H.shape) == 1:
        return HT[0:-1]
    else:
        return HT[:, 0:-1]


def homogenious(coords, value=1):
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if len(coords.shape) == 1:
        H = np.append(coords, value)
    else:
        #H = np.column_stack((coords,np.ones(len(coords))*value))
        #print H
        N, dim = coords.shape
        H = np.empty((N, dim + 1))
        H[:, :-1] = coords
        H[:, -1] = value
    return H


def matrix(t=None,r=None,s=None):
    shape = (0,0)

    if t is not None:
        tM = t_matrix(t)
        shape = tM.shape
    if r is not None:
        rM = r_matrix(r)
        shape = rM.shape
    if s is not None:
        sM = s_matrix(s)
        shape = sM.shape

        
    if t is None:
        tM = i_matrix(shape[0]-1)
    if r is None:
        rM = i_matrix(shape[0]-1)
    if s is None:
        sM = i_matrix(shape[0]-1)
        
    assert tM.shape == shape
    assert rM.shape == shape
    assert sM.shape == shape
        
    return tM * rM * sM
    

def i_matrix(d):
    assert isinstance(d, int)
    return np.matrix(np.identity(d + 1))


def t_matrix(t):
    assert hasattr(t,'__len__') and len(t)>0
    dim = len(t)
    T = np.identity(dim + 1)
    T[0:dim, dim] = t
    return np.matrix(T)


def s_matrix(s):
    assert hasattr(s,'__len__') and len(s)>0
    dim = len(s)
    S = np.identity(dim + 1)
    diag = np.append(s, 1)
    np.fill_diagonal(S, diag)
    return np.matrix(S)


def r_matrix(a):

    if isinstance(a, float):
        R = np.matrix([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])
    else:
        assert hasattr(a,'__getitem__')
        if len(a) == 2:
            raise ValueError('Rotation in 2D requires one angle only.')
        elif len(a) == 3:
            Rx = np.matrix([
                [1, 0, 0, 0],
                [0, np.cos(a[0]), -np.sin(a[0]), 0],
                [0, np.sin(a[0]), np.cos(a[0]), 0],
                [0, 0, 0, 1],
            ])
            Ry = np.matrix([
                [np.cos(a[1]), 0, np.sin(a[1]), 0],
                [0, 1, 0, 0],
                [-np.sin(a[1]), 0, np.cos(a[1]), 0],
                [0, 0, 0, 1],
            ])
            Rz = np.matrix([
                [np.cos(a[2]), -np.sin(a[2]), 0, 0],
                [np.sin(a[2]), np.cos(a[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            R = Rz * Ry * Rx
        else:
            raise ValueError('%i-dimensional rotations are not supported yet'%len(a))
    return R


def addDim(T):
    T = np.matrix(T)
    dim = len(T) + 1
    M = np.eye(dim)
    M[0:(dim - 1), 0:(dim - 1)] = T
    # s=len(T)-1
    # M[0:s,0:s]=T#[0:s,0:s]
    # M[0:s,len(T)]=T[0:s,len(T)-1].flatten()
    return M


def decomposition(T):
    # https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
    # https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix/13165#13165

    T = np.asarray(T)
    assert len(T.shape) == 2 
    assert T.shape[0] == T.shape[1]
    dim = T.shape[0]-1
    
    # translation
    t = np.asarray(T)[:-1,-1]
    
    # scale
    s = distance.norm(np.asarray(T.T))[:-1]
    
    # rotation
    R = T[:-1,:-1]/s
    
    if dim == 2:
        r1 = np.arctan2(R[1,0],R[1,1])
        r2 = np.arctan2(-R[0,1],R[0,0])
        assert np.isclose(r1,r2), 'Rotation angles seem to differ.'
        r = ( r1 + r2 ) * 0.5
    elif dim == 3:
        r_x = np.arctan(R[2, 1] / R[2, 2])
        r_y = -np.arcsin(R[2, 0])
        r_z = np.arctan(R[1, 0] / R[0, 0])
        r = np.array([r_x, r_y, r_z])
    else:
        raise ValueError('Only %s dimensions are not supported jet'%dim)
    
    # determinant
    det = np.linalg.det(T)
    
    return t,r,s,det
   


class LocalSystem(np.matrix, object):

    def __new__(cls, T):
        return np.matrix(T, dtype=float).view(cls)

    @property
    def dim(self):
        return len(self) - 1

    def to_local(self, globalCoords):
        return transform(globalCoords, self)

    def to_global(self, localCoords):
        return transform(localCoords, self, inverse=True)

    def distance(self, coords):
        return distance.norm(self.toLocal(coords)[:, 1:self.dim])

    def PC(self, k):
        assert k <= self.dim
        pc = self[k - 1, :self.dim]
        return np.asarray(pc)[0]

    def explained_variance(self, globalCoords):
        localCoords = self.to_local(globalCoords)
        return np.var(localCoords, axis=0)

    def explained_variance_ratio(self, globalCoords):
        var = self.explained_variance(globalCoords)
        return var / var.sum()

    @property
    def components(self):
        return self[:self.dim, :self.dim]

    @property
    def origin(self):
        return -np.asarray(self[:self.dim, self.dim]).T[0]
