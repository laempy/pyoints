import numpy as np
import itertools as it


class Extent(np.recarray, object):

    def __new__(cls, ext):
        if not isinstance(ext, np.ndarray):
            ext = np.array(ext)
        if len(ext.shape) == 2:
            min_ext = np.amin(ext, axis=0)
            max_ext = np.amax(ext, axis=0)
            ext = np.concatenate((min_ext, max_ext))
        return ext.view(cls)

    @property
    def dim(self):
        return len(self) / 2

    @property
    def ranges(self):
        return self.max_corner - self.min_corner

    @property
    def min_corner(self):
        return self[:self.dim]

    @property
    def max_corner(self):
        return self[self.dim:]

    @property
    def center(self):
        return (self.max_corner + self.min_corner) * 0.5

    @property
    def split(self):
        return self.min_corner, self.max_corner

    @property
    def corners(self):
        genCombs = np.array(list(it.product(range(2), repeat=self.dim)))
        combs = genCombs * self.dim + range(self.dim)
        corners = self[combs]
        if self.dim == 2:
            return corners[(1, 3, 2, 0), :]
            # return corners[(0, 2, 3, 1), :]
        return corners

    def intersects(self, coords, dim=None):
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        if len(coords.shape) == 1:
            coords = np.array([coords])

        n, c_dim = coords.shape
        assert c_dim <= self.dim

        dim = self.dim if dim is None else dim

        min_ext, max_ext = self.split

        # Optimal order
        order = np.argsort(self.ranges[0:dim])

        mask = np.any(
            (np.abs(
                min_ext[order]) < np.inf, np.abs(
                max_ext[order]) < np.inf), axis=0)
        axes = order[mask]

        ids = np.arange(n)
        for axis in axes:

            # Minimum
            mask = coords[:, axis] >= min_ext[axis]
            ids = ids[mask]
            if len(ids) == 0:
                break
            coords = coords[mask, :]

            # Maximum
            mask = coords[:, axis] <= max_ext[axis]
            ids = ids[mask]
            if len(ids) == 0:
                break
            coords = coords[mask, :]

        if n == 1:
            return len(ids) == 1
        return ids