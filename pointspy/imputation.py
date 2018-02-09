import numpy as np
import copy
import sklearn

from . import distance

from misc import *
import matplotlib.pyplot as plt


class KnnImputer:

    def __init__(self, X, missingFlag=99999, k=1, metric='manhattan'):

        self._k = k
        self._c = missingFlag
        self._mu = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0)

        # Prepare data
        xScaled = self.scale(X)
        x = self._prepare(xScaled)

        print 'build tree'
        tic()
        self._ballTree = sklearn.neighbors.BallTree(x, metric=metric)
        toc()

        # self._nearestNeighbours=sklearn.neighbors.NearestNeighbors(n_neighbors=1,metric=metric)
        # self._nearestNeighbours.fit(x)
        self._data = np.copy(X)

        assert np.all(np.any(~np.isnan(X), axis=0)
                      ), 'At least one element per column needed'

    @property
    def data(self):
        return self._data

    @property
    def dim(self):
        return len(self._mu)

    def scale(self, X):
        xScaled = (X - self._mu) / self._std
        return xScaled

    def _prepare(self, xScaled):
        # Doubled features ==> similar distance to NAN values
        nanMask = np.isnan(xScaled)

        x1 = np.copy(xScaled) - self._c
        x1[nanMask] = 0

        x2 = np.copy(xScaled) + self._c
        x2[nanMask] = 0

        return np.hstack((x1, x2))

    def setK(self, k):
        assert isinstance(k), 'number of neighbours k has to be an interger!'
        self._k = k

    @property
    def k(self):
        return self._k

    def validate(self, X, k=None):

        pred = self(X, k=k, full=True)
        NRMSE = np.sqrt(np.nanmean((X - pred)**2, axis=0)) / self._mu

        print 'mean normalized RMSE: %f' % np.nanmean(NRMSE)

        plt.bar(range(self.dim), NRMSE)
        plt.show()

    def __call__(self, X, k=None, full=False):

        pred = np.copy(X)
        for col in range(self.dim):

            if full:
                ids = np.arange(len(X))
            else:
                ids = np.where(np.isnan(X[:, col]))[0]

            if len(ids) > 0:

                pred[ids, col] = self._predictCol(X[ids, :], col, k=k)

                if False:

                    bins = np.linspace(
                        np.nanmin(pred[:, col]), np.nanmax(pred[:, col]), 20)
                    normed = True

                    v = self.data[:, col]
                    v = v[~np.isnan(v)]
                    plt.hist(v, bins, color='green', normed=normed)

                    plt.hist(pred[:, col], bins, color='red',
                             alpha=0.5, normed=normed)

                    plt.show()

        return pred

    def _predictCol(self, X, col, k=None):

        k = self.k if k is None else k

        nanMask = np.isnan(X)

        xScaled = self.scale(X)
        x = self._prepare(xScaled)

        # Ensure same distance to NAN and any other value
        x[:, 0:self.dim][nanMask] = -0.5 * self._c
        x[:, self.dim:][nanMask] = 0.5 * self._c

        # Prevent from selecting a NAN value
        # w=self._c*self.dim
        w = self._c * 2 * self.dim
        x[:, col] = -w
        x[:, self.dim + col] = w

        print 'impute col %i' % col
        if k == 1:
            tic()
            nIds = self._ballTree.query(
                x,
                k=k,
                dualtree=True,
                sort_results=False,
                return_distance=False)
            toc()
            pred = self.data[nIds[:, 0], col]
        else:
            tic()
            dists, nIds = self._ballTree.query(
                x, k=k, dualtree=True, sort_results=False)
            toc()
            weights = distance.IDW(dists)
            pred = np.average(self.data[nIds, col], weights=weights, axis=1)

        if np.any(np.isnan(pred)):
            raise RuntimeError(
                "Unexpected behaviour: missing values in result detected!")

        return pred


class DecisionImputer:

    def __init__(self, X, missingFlag=999999,
                 max_classes=20,
                 base_regressor=sklearn.tree.DecisionTreeRegressor(),
                 base_classifier=sklearn.tree.DecisionTreeClassifier(
                     class_weight="balanced"),
                 # base_regressor=sklearn.ensemble.RandomForestRegressor(n_jobs=-1),
                 # base_classifier=sklearn.ensemble.RandomForestClassifier(n_jobs=-1,class_weight="balanced")
                 ):

        assert np.all(np.any(~np.isnan(X), axis=0)
                      ), 'At least one element per column needed'

        self._c = missingFlag
        self._dim = X.shape[1]
        self._base_regressor = copy.copy(base_regressor)
        self._base_classifier = copy.copy(base_classifier)
        self._estimators = {}
        self._max_classes = max_classes

        self._data = X

    @property
    def data(self):
        return self._data

    @property
    def dim(self):
        return self._dim

    def _fitCol(self, X, col, plot=False):

        colMask = np.ones(self.dim, dtype=bool)
        colMask[col] = False

        data = np.copy(X)[~np.isnan(X[:, col]), :]

        # features
        x = data[:, colMask]
        x[np.isnan(x)] = self._c

        # values
        y = data[:, ~colMask]
        y = np.array(y.tolist())

        # check if the values are discrete
        if (np.array(y, dtype=int) == y).sum() == len(
                y) and len(np.unique(y)) < self._max_classes:
            estimator = copy.copy(self._base_classifier)
        else:
            estimator = copy.copy(self._base_regressor)
        print 'impute column %i with %s' % (col, type(estimator))
        estimator.fit(x, y)

        if plot:
            pred = estimator.predict(x)
            bins = np.linspace(np.nanmin(pred), np.nanmax(pred), 20)
            normed = True

            plt.hist(y, bins, color='green', normed=normed)
            plt.hist(pred, bins, color='red', alpha=0.5, normed=normed)

            plt.show()

        self._estimators[col] = estimator

    def __call__(self, X, full=False, plot=False):

        pred = np.copy(X)
        for col in range(self.dim):

            if full:
                ids = np.arange(len(X))
            else:
                ids = np.where(np.isnan(X[:, col]))[0]

            if len(ids) > 0:

                # Just fit when necessary
                if col not in self._estimators:
                    self._fitCol(self._data, col, plot=plot)

                pred[ids, col] = self._predictCol(X[ids, :], col)

                if plot:
                    bins = np.linspace(
                        np.nanmin(pred[:, col]), np.nanmax(pred[:, col]), 20)
                    normed = True

                    v = self.data[:, col]
                    mask = ~np.isnan(v)
                    plt.hist(v[mask], bins, color='green', normed=normed)
                    plt.hist(pred[:, col], bins, color='red',
                             alpha=0.5, normed=normed)

                    plt.show()

        return pred

    def _predictCol(self, X, col):

        assert col in self._estimators, 'Column %i not fitted yet' % col

        colMask = np.ones(self.dim, dtype=bool)
        colMask[col] = False

        data = np.copy(X)
        x = data[:, colMask]
        x[np.isnan(x)] = self._c

        estimator = self._estimators[col]
        pred = estimator.predict(x)

        return pred

    def validate(self, X):

        pred = self(X, full=True, plot=False)
        NRMSE = np.sqrt(np.nanmean((X - pred)**2, axis=0)) / \
            np.nanmean(X, axis=0)

        print 'mean normalized RMSE: %f' % np.nanmean(NRMSE)

        plt.bar(range(self.dim), NRMSE)
        plt.show()

        for col in range(self.dim):
            print col
            bins = np.linspace(
                np.nanmin(pred[:, col]), np.nanmax(pred[:, col]), 20)
            normed = False

            mask = ~np.isnan(X[:, col])
            plt.hist(X[mask, col], bins, color='green', normed=normed)
            plt.hist(pred[mask, col], bins, color='red',
                     alpha=0.5, normed=normed)
            plt.hist(pred[~mask, col], bins, color='blue',
                     alpha=0.5, normed=normed)

            plt.show()
