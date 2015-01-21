#cython: boundscheck=False, wraparound=False, cdivision=True

"""TSTE_sideinfo: t-Distributed Stochastic Triplet Embedding, with
side information.

Original MATLAB implementation of tSTE and tSNE: (C) Laurens van der Maaten, 2012, Delft University of Technology

Also uses implementation of t_SNE from scikit-learn. (C) Alexander
Fabisch -- <afabisch@informatik.uni-bremen.de>

Curator: Michael Wilber <mjw285@cornell.eu>

"""

cimport numpy as cnp
import numpy as np
from libc.math cimport log
cimport cython.parallel
cimport openmp

cpdef tste_grad(npX,
                int N,
                int no_dims,
                long [:, ::1] triplets,
                double alpha,
):
    """Compute the cost function and gradient update of t-STE.

    Note that this is a bit different from t-STE's paper! Our utility
    function (NOT cost!) is how many "incorrect" triplets are
    satisfied. We then want to minimize the number of unsatisfied
    triplets.

    The gradient seems OK --- see check_gradient.py here.

    Parameters
    ----------
    npX : The embedding so far. Should have shape (N, no_dims)
    N : Number of points to embed, should be equal to len(npX) I think (oops)
    no_dims : Dimensionality of the embedding
    triplets : Crowdsourced triplets, with shape (N, 3). List of
               (a,b,c)-tuples where a is closer to b than a is to c
    alpha : Degrees of freedom in the Student-T kernel.
    """
    cdef long[:] triplets_A = triplets[:,0]
    cdef long[:] triplets_B = triplets[:,1]
    cdef long[:] triplets_C = triplets[:,2]
    cdef int i,t,j,k
    cdef double[:, ::1] X = npX
    cdef unsigned int no_triplets = len(triplets)
    cdef double P = 0
    cdef double C = 0
    cdef double A_to_B, A_to_C, const

    # We don't perform regularization in this function.

    cdef double[::1] sum_X = np.zeros((N,), dtype='float64')
    cdef double[:, ::1] K = np.zeros((N, N), dtype='float64')
    cdef double[:, ::1] Q = np.zeros((N, N), dtype='float64')
    npdC = np.zeros((N, no_dims), 'float64')
    cdef double[:, ::1] dC = npdC
    cdef double[:, :, ::1] dC_part = np.zeros((no_triplets, no_dims, 3), 'float64')

    # Compute student-T kernel for each point
    # i,j range over points; k ranges over dims
    with nogil:
        for i in xrange(N):
            sum_X[i] = 0
            for k in xrange(no_dims):
                # Squared norm
                sum_X[i] += X[i,k]*X[i,k]
        for i in cython.parallel.prange(N):
            for j in xrange(N):
                K[i,j] = sum_X[i] + sum_X[j]
                for k in xrange(no_dims):
                    K[i,j] += -2 * X[i,k]*X[j,k]
                Q[i,j] = (1 + K[i,j] / alpha) ** -1
                K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)
                # Now, K[i,j] = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
                # which is exactly the numerator of p_{i,j} in the lower right of
                # t-STE paper page 3.
                # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab

                # (Note however that we're flipping the long and short
                # edge, since this should be unsatisfied)

        # Compute probability (or log-prob) for each triplet
        for t in cython.parallel.prange(no_triplets):
            P = K[triplets_A[t], triplets_C[t]] / (
                K[triplets_A[t],triplets_B[t]] +
                K[triplets_A[t],triplets_C[t]])
            # This is a mirror image of the equation in the
            # lower-right of page 3 of the t-STE paper. Note that this
            # works because K is some reciprocal of the distance, so
            # I'm convinced this is correct.
            C += P
            # The probability that triplet (a,b,c) is UNSATISFIED.
            # (We want to MINIMIZE C)

            for i in xrange(no_dims):
                # For i = each dimension to use:
                # Calculate the gradient of *this triplet* on its points.
                const = (alpha+1) / alpha
                A_to_B = ((1 - P) *
                          Q[triplets_A[t],triplets_B[t]] *
                          (X[triplets_A[t], i] - X[triplets_B[t], i]))
                A_to_C = ((1 - P) *
                          Q[triplets_A[t],triplets_C[t]] *
                          (X[triplets_A[t], i] - X[triplets_C[t], i]))

                # Problem: Since this is a parallel for loop, we can't
                # accumulate everything at once. Race conditions.
                # So I calculate it once here:
                dC_part[t, i, 0] = const * P * (A_to_B - A_to_C)
                dC_part[t, i, 1] = const * P * (-A_to_B)
                dC_part[t, i, 2] = const * P * (A_to_C)

                # This is like 'use_log=False', which doesn't make
                # sense in this inverted "minimize number of
                # unsatisfied triplets" formulation.

        # ...and then accumulate:
        for n in xrange(N):
            for i in xrange(no_dims):
                dC[n, i] = 0
        for t in xrange(no_triplets):
            for i in xrange(no_dims):
                dC[triplets_A[t], i] += dC_part[t, i, 0]
                dC[triplets_B[t], i] += dC_part[t, i, 1]
                dC[triplets_C[t], i] += dC_part[t, i, 2]
        # I'm not performing regularization here!
        # for n in xrange(N):
        #     for i in xrange(no_dims):
        #         # The 2*lamb*npx is for regularization: derivative of L2 norm
        #         dC[n,i] = (dC[n,i])
    return C, npdC

def tste(triplets,
         no_dims=2,
         alpha=None,
         verbose=True,
         max_iter=1000,
         save_each_iteration=False,
         initial_X=None,
         static_points=np.array([]),
         num_threads=None,
         use_log=False,
):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    Parameters:

    triplets: An Nx3 integer array of object indices. Each row is a
              triplet; first column is the 'reference', second column
              is the 'near edge', and third column is the 'far edge'.
    no_dims:  Number of dimensions in final embedding. High-dimensional
              embeddings are much easier to satisfy (lower training
              error), but may capture less information.
    alpha:    Degrees of freedom in student T kernel. Default is no_dims-1.
              Considered "black magic"; roughly, how much of an impact
              badly satisfying points have on the gradient calculation.
    verbose:  Prints log messages every 10 iterations
    save_each_iteration: When true, will save intermediate results to
              a list so you can watch it converge.
    initial_X: The initial set of points to use. Normally distributed if unset.
    num_threads: Parallelism.

    """
    if num_threads is None:
        num_threads = openmp.omp_get_num_procs()
    openmp.omp_set_num_threads(num_threads)

    if alpha is None:
        alpha = no_dims-1

    N = np.max(triplets) + 1
    assert -1 not in triplets

    n_triplets = len(triplets)

    # Initialize some variables
    if initial_X is None:
        X = np.random.randn(N, no_dims) * 0.0001
    else:
        X = initial_X


    # Cheating!
    from sklearn.manifold.t_sne import _gradient_descent
    saved_iterations = []
    def work(x):
        saved_iterations.append(x.copy().reshape(N, no_dims))
        C,dC = tste_grad(x.reshape(N, no_dims), N, no_dims, triplets, alpha)

        X=x.reshape(N, no_dims)
        sum_X = np.sum(X**2, axis=1)
        D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
        # ^ D = squared Euclidean distance?
        no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
        print 'Cost is ',C,', number of constraints: ', (float(no_viol) / n_triplets)

        return C, dC.ravel()
    params, iter, it = _gradient_descent(
        work,
        X.ravel(),
        it=0,
        n_iter=max_iter,
        n_iter_without_progress=5,
        momentum=0.0,
        learning_rate = (float(2.0) / n_triplets * N),
        min_gain = 1e-5,
        min_grad_norm = 1e-7, # Abort when less
        min_error_diff = 1e-7,
        verbose=5,
    )

    if save_each_iteration:
        return params.reshape(N, no_dims), saved_iterations
    else:
        return params.reshape(N, no_dims)

    #C = np.Inf
    #tol = 1e-7              # convergence tolerance
    #eta = 2.                # learning rate
    #best_C = np.Inf         # best error obtained so far
    #best_X = X              # best embedding found so far
    #iteration_Xs = []       # for debugging ;) *shhhh*

    ## Perform main iterations
    #iter = 0; no_incr = 0;
    #while iter < max_iter and no_incr < 5:
    #    old_C = C
    #    # Calculate gradient descent and cost
    #    C, G = tste_grad(X, N, no_dims, triplets, alpha)
    #    X = X - (float(eta) / n_triplets * N) * G

    #    if C < best_C:
    #        best_C = C
    #        best_X = X

    #    # Perform gradient update
    #    if save_each_iteration:
    #        iteration_Xs.append(X.copy())

    #    if len(static_points):
    #        X[static_points] = initial_X[static_points]

    #    # Update learning rate
    #    if old_C > C + tol:
    #        no_incr = 0
    #        eta *= 1.01
    #    else:
    #        no_incr = no_incr + 1
    #        eta *= 0.5

    #    # Print out progress
    #    iter += 1
    #    if verbose and iter%10==0:
    #        # These are Euclidean distances:
    #        sum_X = np.sum(X**2, axis=1)
    #        D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
    #        # ^ D = squared Euclidean distance?
    #        no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
    #        print "Iteration ",iter, ' error is ',C,', number of constraints: ', (float(no_viol) / n_triplets)

    #if save_each_iteration:
    #    return iteration_Xs
    #return best_X
