import numpy as np
from sklearn import ensemble
from sklearn import preprocessing

from . import npTools
from . import imputation
from .misc import *

import matplotlib.pyplot as plt


class CART:

    def checkFitted(self):
        if not hasattr(self, 'feature_importances_'):
            raise NotFittedError('Estimator not fitted yet.')

    @property
    def pDtype(self):
        self.checkFitted()
        return self._pDtype

    @property
    def fDtype(self):
        self.checkFitted()
        return self._fDtype

    @property
    def outDtype(self):
        raise NotImplementedError()


class RandomForestClassifier(CART, ensemble.RandomForestClassifier):

    def fit(self, X, y, imputation=lambda x: x, **kwargs):

        self._pDtype = y.dtype
        self._fDtype = X.dtype
        Y = np.array(zip(*y)).T
        if Y.shape[1] == 1:
            Y = np.ravel(Y)

        #print [X[name] for name in X.dtype.names]
        x = npTools.mergeColumns(X, dtype=float)

        # Imputation
        self._imputationFunction = imputation
        x = self._imputationFunction(x)

        tic()
        super(ensemble.RandomForestClassifier, self).fit(x, Y, **kwargs)
        toc()

        # Define labels
        self._labels = {}
        for name in self.pDtype.names:
            labels = np.unique(y[name])
            order = np.argsort(labels.astype(str))  # order is required
            self._labels[name] = labels[order]

    @property
    def labels(self):
        return self._labels
        # labels=self.classes_
        # if len(self.pDtype.names)==1:
        #    labels=[self.classes_]
        # return dict([(name,label.astype(self.pDtype[name])) for (name,label)
        # in zip(self.pDtype.names,labels)])

    @property
    def outDtype(self):
        return np.dtype([(name, [('p', float), ('pred', self.pDtype[name])])
                         for name in self.pDtype.names])

    def predict(self, X):
        self.checkFitted()

        # Preparation
        x = npTools.mergeColumns(X, dtype=float)
        x = self._imputationFunction(x)

        tic()
        rawProbs = super(self.__class__, self).predict_proba(x)
        toc()
        if len(self.pDtype.names) == 1:
            rawProbs = [rawProbs]

        pred = np.recarray(rawProbs[0].shape[0], dtype=self.outDtype)

        for col, name in enumerate(self.pDtype.names):
            labels = self.labels[name]
            pMask = np.argmax(rawProbs[col], axis=1)
            pred[name]['pred'] = labels[pMask]

            # Boolean output or multi-class?
            if len(labels) == 2 and True in labels:
                i = np.where(labels)[0][0]
                pred[name]['p'] = rawProbs[col][:, i]
            else:
                pred[name]['p'] = rawProbs[col].max(1)

        return pred


class RandomForestRegressor(CART, ensemble.RandomForestRegressor):

    def fit(self, X, y, **kwargs):
        self._pDtype = y.dtype
        self._fDtype = X.dtype
        x = npTools.fuse(X, dtype=float)
        super(
            ensemble.RandomForestRegressor,
            self).fit(
            x,
            y.tolist(),
            **kwargs)

    def predict(self, X):
        self.checkFitted()

        x = npTools.mergeColumns(X, dtype=float)
        rawPred = super(self.__class__, self).predict(x)
        if len(self.pDtype.names) == 1:
            rawPred = np.array([rawPred]).T

        predDict = {}
        for i, name in enumerate(self.pDtype.names):
            predDict[name] = rawPred[:, i]
        pred = npTools.recarray(predDict, dtype=self.outDtype)

        return pred

    @property
    def outDtype(self):
        return self.pDtype
