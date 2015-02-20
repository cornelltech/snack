#!/usr/bin/env python2

from sklearn.metrics import euclidean_distances
from sklearn.manifold import t_sne
import numpy as np
import _snack as snack

for i in xrange(10):
    X = np.random.randn(1000, 2) * 10
    params = X.ravel()
    D = euclidean_distances(X)

    probs0 = t_sne._joint_probabilities(D, 30, False)
    probs1 = snack.my_joint_probabilities(D, 30, False)
    c1,grad1 = t_sne._kl_divergence(params, probs0, 1.0, len(X), 2)
    c2,grad2 = snack.my_kl_divergence(params, probs1, 1.0, len(X), 2.0)
    print "Test", i
    print "Difference norm:", np.linalg.norm(probs0 - probs1)
    print "Difference norm:", np.linalg.norm(grad1 - grad2)
    print "Difference norm:", c1-c2
