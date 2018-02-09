import numpy as np
import distance


def transform(coords, T, inverse=False):

    if inverse:
        T = np.linalg.inv(T)
    T = np.asarray(T)

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


def i_matrix(d):
    return np.matrix(np.identity(d + 1))


def t_matrix(t):
    dim = len(t)
    T = np.identity(dim + 1)
    T[0:dim, dim] = t
    return np.matrix(T)


def s_matrix(s):
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
        raise ValueError(str(len(a)) +
                         '-dimensional rotations are not supported yet')
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


def decomposition(R):
    assert R.shape[0] == 4 and R.shape[1] == 4, 'Only 3 dimensions are supported jet'
    angle_x = np.arctan(R[2, 1] / R[2, 2])
    angle_y = -np.arcsin(R[2, 0])
    angle_z = np.arctan(R[1, 0] / R[0, 0])
    angles = np.array([angle_x, angle_y, angle_z])
    shift = np.asarray(R[0:3, 3])
    det = np.linalg.det(R)
    return shift, angles, det


class LocalSystem(np.matrix, object):

    def __new__(cls, T):
        return np.matrix(T, dtype=float).view(cls)

    @property
    def dim(self):
        return len(self) - 1

    def toLocal(self, globalCoords):
        return transform(globalCoords, self)

    def toGlobal(self, localCoords):
        return transform(localCoords, self, inverse=True)

    def distance(self, coords):
        return distance.norm(self.toLocal(coords)[:, 1:self.dim])

    def PC(self, k):
        assert k <= self.dim
        pc = self[k - 1, :self.dim]
        return np.asarray(pc)[0]

    def explainedVariance(self, globalCoords):
        localCoords = self.toLocal(globalCoords)
        return np.var(localCoords, axis=0)

    def explainedVarianceRatio(self, globalCoords):
        var = self.explainedVariance(globalCoords)
        return var / var.sum()

    @property
    def components(self):
        return self[:self.dim, :self.dim]

    @property
    def origin(self):
        return -np.asarray(self[:self.dim, self.dim]).T[0]
