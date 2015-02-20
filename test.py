#!/usr/bin/env python2

from sklearn.metrics import euclidean_distances
from sklearn.manifold import t_sne
import numpy as np
import _snack as snack

for i in xrange(10):
    X = np.random.randn(1000, 2) * 10
    params = X.ravel()
    D = euclidean_distances(X)

    probs1 = t_sne._joint_probabilities(D, 30, False)
    probs2 = snack.my_joint_probabilities(D, 30, False)
    c1,grad1 = t_sne._kl_divergence(params, probs1, 1.0, len(X), 2)
    c2,grad2 = snack.my_kl_divergence(params, probs1, 1.0, len(X), 2.0)
    print "Test", i
    print "Difference norm:", np.linalg.norm(probs1 - probs2)
    print "Difference norm:", np.linalg.norm(grad1 - grad2)
    print "Difference norm:", c1-c2

    assert np.allclose(probs1, probs2)
    assert np.allclose(grad1, grad2)
    assert np.allclose(c1, c2)
