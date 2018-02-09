import numpy as np


class NormalResiduals:

    def __init__(self, sd):
        self.sd = sd
        self.dim = 1 if isinstance(sd, float) else len(sd)
        assert self.dim == 1 or self.dim == 2 or self.dim == 3, 'Wrong dimension'

    def __call__(self, n=1):

        r = np.random.normal(size=n)
        if self.dim == 1:
            return r
        if self.dim == 2:
            alpha = np.random.random(n) * np.pi
            x = r * np.sin(alpha) * self.sd[0]
            y = r * np.cos(alpha) * self.sd[1]
            return np.array((x, y)).T
        elif self.dim == 3:
            alpha = np.random.random(n) * np.pi
            betha = np.random.random(n) * np.pi
            x = r * np.sin(alpha) * np.cos(betha) * self.sd[0]
            y = r * np.sin(alpha) * np.sin(betha) * self.sd[1]
            z = r * np.cos(alpha) * self.sd[2]
            return np.array((x, y, z)).T


class GammaResiduals:

    def __init__(self, sGammaVp, sGammaVs):
        self.sGammaVp = sGammaVp
        self.sGammaVs = sGammaVs

    def __call__(self, n=1):
        return np.random.gamma(self.sGammaVp, scale=self.sGammaVs, size=n)
