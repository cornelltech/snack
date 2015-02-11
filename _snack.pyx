#cython: boundscheck=False, wraparound=False, cdivision=True

"""SNaCK embedding: Stochastic Neighbor and Crowd Kernel embedding.

Works by stapling together the t-SNE (t-distributed Stochastic Neighbor Embedding) and t-STE (t-distributed Stochastic Triplet Embedding) loss functions.

Original MATLAB implementation of tSTE and tSNE: (C) Laurens van der Maaten, 2012, Delft University of Technology

Also uses implementation of t_SNE from scikit-learn. (C) Alexander
Fabisch -- <afabisch@informatik.uni-bremen.de>

Curator: Michael Wilber <mjw285@cornell.eu>

"""

cimport cython
cimport numpy as np
cimport cython.parallel
cimport openmp

import numpy as np
import tempfile
import os
import gc

from libc cimport math
from scipy import linalg
from sklearn.manifold.t_sne import _gradient_descent, _kl_divergence
from sklearn.manifold import _utils
from scipy.spatial.distance import squareform

cdef extern from "numpy/npy_math.h":
    float NPY_INFINITY

cdef double EPSILON_DBL = 1e-7
cdef double PERPLEXITY_TOLERANCE = 1e-5
cdef double MACHINE_EPSILON = np.finfo(np.double).eps

TEMP_FOLDER = "/tmp/"

def set_temp_folder(fdr):
    global TEMP_FOLDER
    TEMP_FOLDER = fdr

def mmapped_zeros(shape, dtype):
    """Return a array that is automatically persisted to disk. Useful for
    machines without a lot of memory or swap.

    The resource is deleted when the array is garbage-collected.
    """
    global TEMP_FOLDER
    # In theory, all this should be equivalent to
    return np.zeros(shape=shape, dtype=dtype)
    # but it should magically work when your machine doesn't have memory but does have hard disk.
    # To see how many memmaps we have active (watch your gc!),
    # run pmap $PID | grep .mmap
    (_, filename) = tempfile.mkstemp(suffix='.mmap',
                                     dir=TEMP_FOLDER,
                                     )
    os.unlink(filename)
    A = np.memmap(filename, shape=shape, dtype=dtype, mode='w+')
    os.unlink(filename)
    return A

def my_joint_probabilities(
        np.ndarray[np.double_t, ndim=2] distances,
        desired_perplexity,
        verbose):
    """Compute joint probabilities p_ij from distances.

    Just like t_sne._joint_probabilities, but this one avoids
    unnecessary allocations.

    Parameters
    ----------
    distances : array, shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity

    # Make a temporary mmapped file, for memory safety
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose)
    cdef double[:, ::1] cPview = conditional_P

    # Symmetricise
    cdef int n_samples = len(conditional_P)
    cdef int i,j
    for i in xrange(n_samples):
        for j in xrange(i+1, n_samples):
            cPview[i,j] = cPview[i,j] + cPview[j,i]
            cPview[j,i] = cPview[i,j]
    sum_P = np.maximum(np.sum(conditional_P), MACHINE_EPSILON)

    # Turn into probabilities
    npP = squareform(conditional_P)
    cdef double[::1] P = npP
    del conditional_P, cPview
    for i in xrange(len(P)):
        P[i] = P[i] / sum_P
        if P[i] < MACHINE_EPSILON:
            P[i] = MACHINE_EPSILON
    return npP


def my_kl_divergence(
        params,
        double [::1] P,
        double alpha,
        int n_samples,
        int n_components):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Just like sklearn.manifold.t_sne._kl_divergence, but this version
    avoids spurious allocations.

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.

    alpha : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.

    """
    cdef double[:,::1] X_embedded = params.reshape(n_samples, n_components)

    # Step 1: Calculate sum(n)
    cdef double sum_n = 0.0
    cdef int i,j,k
    cdef double dist
    for i in xrange(n_samples):
        for j in xrange(i+1, n_samples):
            dist = 0
            for k in xrange(n_components):
                dist += ((X_embedded[i,k]-X_embedded[j,k]) *
                         (X_embedded[i,k]-X_embedded[j,k]))
            sum_n = sum_n + ((1+dist)/alpha) ** ((alpha+1.0) / -2.0)

    # Step 2: Calculate Q, which is the Student's t-distribution here.
    # (Trading off memory use for speed here)
    # I'm also inlining the calculation for the gradient here.
    npgrad = np.zeros((n_samples, n_components), dtype='double')
    cdef double[:,::1] grad = npgrad
    cdef double kl_divergence = 0.0
    cdef double Q = 0.0
    cdef int dim_idx = 0
    for i in xrange(n_samples):
        for j in xrange(i+1, n_samples):
            dist = 0
            for k in xrange(n_components):
                dist += ((X_embedded[i,k]-X_embedded[j,k]) *
                         (X_embedded[i,k]-X_embedded[j,k]))

            # What's this Q?
            Q = ((1+dist)/alpha) ** ((alpha+1.0) / -2.0)
            Q = max(Q/ (2.0 * sum_n), MACHINE_EPSILON)

            kl_divergence += 2.0 * P[dim_idx] * math.log(P[dim_idx] / Q)

            dim_idx += 1

        # Inline: Add j's result to grad[i]
        for j in xrange(n_samples):
            if i==j: continue
            # verified in my notebook :S .... but I do hate it!
            # see "2015-02-06 Make tSNE part use less memory"
            dim_idx = n_samples*min(i,j) - (min(i,j)*(min(i,j) + 3)/2) + max(i,j) - 1
            for k in xrange(n_components):
                grad[i,k] += (
                    # PQd
                    (P[dim_idx] - Q) * ((1+dist)/alpha) ** ((alpha+1.0) / -2.0)
                    # Difference between this point and every other point
                    * (X_embedded[i,k] - X_embedded[j, k])
                ) * (2.0 * (alpha + 1.0) / alpha) # this is c

    return kl_divergence, npgrad.ravel()


cpdef tste_grad(npX,
                int N,
                int no_dims,
                long [:, ::1] triplets,
                double alpha,
                double[:,::1] K,
                double[:,::1] Q,
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
    # Don't need to reinitialize K, Q because they're initialized below in the loop.
    assert K.shape[0] == N; assert K.shape[1] == N
    assert Q.shape[0] == N; assert Q.shape[1] == N
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


def snack_embed(triplets,
                distances,
                perplexity=30,
                no_dims=2,
                contrib_cost_triplets=1.0,
                contrib_cost_tsne=1.0,
                alpha=None,
                verbose=True,
                max_iter=300,
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
    P = my_joint_probabilities(distances, perplexity, verbose=10 if verbose else 0)

    # Normalize triplet cost by the number of triplets that we have
    contrib_cost_triplets = contrib_cost_triplets*(2.0 / float(n_triplets) * float(N))

    cdef double[:,::1] K = mmapped_zeros((N, N), dtype='float64')
    cdef double[:,::1] Q = mmapped_zeros((N, N), dtype='float64')

    def grad(x):
        # t-STE
        C1,dC1 = tste_grad(x.reshape(N, no_dims), N, no_dims, triplets, alpha,
                           K, Q)
        dC1 = dC1.ravel()

        # t-SNE
        C2,dC2 = my_kl_divergence(x, P, alpha, N, no_dims)

        C = contrib_cost_triplets*C1 + contrib_cost_tsne*C2
        dC = contrib_cost_triplets*dC1 + contrib_cost_tsne*dC2

        # Calculate and report # of triplet violations
        X=x.reshape(N, no_dims)
        sum_X = np.sum(X**2, axis=1)
        no_viol = -1
        if verbose:
            #D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
            ## ^ D = squared Euclidean distance?
            #no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
            print 'Cost is ',C #,', number of constraints: ', (float(no_viol) / n_triplets)

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
        verbose=5 if verbose else 0,
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
        verbose=5 if verbose else 0,
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
        verbose=5 if verbose else 0,
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
