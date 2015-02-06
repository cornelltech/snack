#!/usr/bin/env python2

import _snack as cy_tste # whaaat???
import numpy as np

for N in [10, 100, 1000]:
    for ndim in [2, 5, 10]:
        for ntriplets in [1000]:
            for use_log in [True, False]:
                print N,ndim,ntriplets,use_log
                X = np.random.randn(N, ndim)
                triplets = (N*np.random.rand(ntriplets,3)).astype('int')

                #_G = np.zeros((N, ndim), 'float64')
                #_sum_X = np.zeros((N,), dtype='float64')
                #_K = np.zeros((N, N), dtype='float64')
                #_Q = np.zeros((N, N), dtype='float64')
                #_dCdt= np.zeros((ntriplets, ndim, 3), 'float64')

                def run_tste(Xsofar):
                    C, G = cy_tste.tste_grad(
                        Xsofar, N, ndim, triplets, (ndim-1),
                        #use_log,
                        #_sum_X, _K, _Q, _G, _dCdt,
                        )
                    return C, G#(_G.copy())

                _, dC = run_tste(X)
                observed_dC = np.zeros((N, ndim))
                for point in xrange(N):
                    for dim in xrange(ndim):
                        h = 0.0000001
                        # Does nudging this point do anything?
                        dx = np.zeros(X.shape)
                        dx[point, dim] += h
                        C2, _ = run_tste(X + dx)
                        C1, _ = run_tste(X - dx)
                        observed_dC[point,dim] = (C2 - C1) / (2*h)

                print "Norm:", (np.linalg.norm(observed_dC - dC) / np.linalg.norm(observed_dC + dC))


print """
All norms should be super small; eg. less than 1e-5
"""
