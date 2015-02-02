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

from scipy import linalg

from sklearn.manifold.t_sne import _gradient_descent, _kl_divergence, _joint_probabilities

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
    cdef double[::1] sum_X = np.zeros((N,), dtype='float64')
    cdef double[:, ::1] K = np.zeros((N, N), dtype='float64')
    cdef double[:, ::1] Q = np.zeros((N, N), dtype='float64')
    npdC = np.zeros((N, no_dims), 'float64')
    cdef double[:, ::1] dC = npdC
    cdef double[:, :, ::1] dC_part = np.zeros((no_triplets, no_dims, 3), 'float64')

    # We don't perform L2 regularization, unlike original tSTE

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

        # Compute probability (or log-prob) for each triplet Note that
        # each of these probabilities are FLIPPED; ie. this is the
        # probability that the triplet (a,b,c) is VIOLATED
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

# def debug_costs(samples, triplets, X_embedded, perplexity, which):
#     # tSNE perplexity calculation
#     P = _joint_probabilities(distances, perplexity, verbose=10)
#     for i in samples:

#     n = pdist(X_embedded, "sqeuclidean")
#     n += 1.
#     n /= alpha
#     n **= (alpha + 1.0) / -2.0
#     Q = np.maximum(n / (2.0 * np.sum(n)), MACHINE_EPSILON)

#     # Optimization trick below: np.dot(x, y) is faster than
#     # np.sum(x * y) because it calls BLAS

#     # Objective: C (Kullback-Leibler divergence of P and Q)
#     kl_divergence = 2.0 * np.dot(P, np.log(P / Q))


def frankentriplet_tsne(triplets,
                        distances,
                        perplexity=30,
                        no_dims=2,
                        contrib_cost_triplets=1.0,
                        contrib_cost_tsne=1.0,
                        alpha=None,
                        verbose=True,
                        max_iter=1000,
                        initial_X=None,
                        static_points=np.array([]),
                        num_threads=None,
                        use_log=False,
                        each_function=False,
):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    Parameters:

    triplets: An Nx3 integer array of object indices. Each row is a
              triplet; first column is the 'reference', second column
              is the 'near edge', and third column is the 'far edge'.
    distances: A square distance matrix for t-SNE.
    no_dims:  Number of dimensions in final embedding. High-dimensional
              embeddings are much easier to satisfy (lower training
              error), but may capture less information.
    alpha:    Degrees of freedom in student T kernel. Default is no_dims-1.
              Considered "black magic"; roughly, how much of an impact
              badly satisfying points have on the gradient calculation.
    verbose:  Prints log messages every 10 iterations
    initial_X: The initial set of points to use. Normally distributed if unset.
    num_threads: Parallelism.
    each_function: A function that is called for each gradient update

    """
    if num_threads is None:
        num_threads = openmp.omp_get_num_procs()
    openmp.omp_set_num_threads(num_threads)

    if alpha is None:
        alpha = no_dims-1

    N = np.max(triplets) + 1
    assert -1 not in triplets
    assert 0 in triplets, ("This is not Matlab. Indices should be 0-indexed. "+
                           "Remove this assertion if you really want point 0 to "+
                           "have no gradient.")

    n_triplets = len(triplets)

    # Initialize some variables
    if initial_X is None:
        X = np.random.randn(N, no_dims) * 0.0001
    else:
        X = initial_X

    # tSNE perplexity calculation
    P = _joint_probabilities(distances, perplexity, verbose=10)

    def grad(x):
        # t-STE
        C1,dC1 = tste_grad(x.reshape(N, no_dims), N, no_dims, triplets, alpha)
        dC1 = dC1.ravel()

        # t-SNE
        C2,dC2 = _kl_divergence(x, P, alpha, N, no_dims)

        C = contrib_cost_triplets*C1 + contrib_cost_tsne*C2
        dC = contrib_cost_triplets*dC1 + contrib_cost_tsne*dC2

        # Calculate and report # of triplet violations
        X=x.reshape(N, no_dims)
        sum_X = np.sum(X**2, axis=1)
        D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
        # ^ D = squared Euclidean distance?
        no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
        print 'Cost is ',C,', number of constraints: ', (float(no_viol) / n_triplets)

        if each_function:
            each_function(x.copy().reshape(N,no_dims),
                          linalg.norm(dC1),
                          linalg.norm(dC2),
                          contrib_cost_triplets*linalg.norm(dC1),
                          contrib_cost_tsne*linalg.norm(dC2),
                          no_viol,
                          C,
            )

        return C, dC

    # Early exaggeration
    EARLY_EXAGGERATION = 4.0
    P *= EARLY_EXAGGERATION
    params, iter, it = _gradient_descent(
        grad,
        X.ravel(),
        it=0,
        n_iter=50,
        n_iter_without_progress=300,
        momentum=0.5,
        learning_rate = 1.0, # Chosen by the caller!
        min_gain = 1e-5,
        min_grad_norm = 1e-7, # Abort when less
        min_error_diff = 1e-7,
        verbose=5,
    )
    # Second stage: More momentum
    params, iter, it = _gradient_descent(
        grad,
        params,
        it=it+1,
        # n_iter=max_iter,
        n_iter=100,
        n_iter_without_progress=300,
        momentum=0.8,
        learning_rate = 1.0, #(float(2.0) / n_triplets * N),
        min_gain = 1e-5,
        min_grad_norm = 1e-7, # Abort when less
        min_error_diff = 1e-7,
        verbose=5,
    )
    # Undo early exaggeration
    P /= EARLY_EXAGGERATION
    params, iter, it = _gradient_descent(
        grad,
        params,
        it=it+1,
        n_iter=max_iter,
        n_iter_without_progress=300,
        momentum=0.8,
        learning_rate = 1.0, #(float(2.0) / n_triplets * N),
        min_gain = 1e-5,
        min_grad_norm = 1e-7, # Abort when less
        min_error_diff = 1e-7,
        verbose=5,
    )
    # params, iter, it = _gradient_descent(
    #     work,
    #     X.ravel(),
    #     it=it+1,
    #     n_iter=max_iter,
    #     n_iter_without_progress=300,
    #     momentum=0.5,
    #     learning_rate = 1.0, #(float(2.0) / n_triplets * N),
    #     min_gain = 1e-5,
    #     min_grad_norm = 1e-7, # Abort when less
    #     min_error_diff = 1e-7,
    #     verbose=5,
    # )

    return params.reshape(N, no_dims)
