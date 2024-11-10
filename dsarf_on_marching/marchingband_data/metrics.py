import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def compute_regime_labeling_accuracy(
    estimated_regime_seq,
    true_regime_seq,
) -> float:
    """
    Due to label-switching, need to find first find the permutation that best matches the truth
    in order to compute accuracy.
    """
    # Convert types in case we have jax arrays with int32s.  This is necessary because
    # `find_permutation` does a type checking and assumes the type is int, not int32.
    estimated_regime_seq = np.asarray(estimated_regime_seq, dtype=int)
    true_regime_seq = np.asarray(true_regime_seq, dtype=int)

    # Rk: The `find_permutation` function requires numpy arrays
    perm_of_estimated = find_permutation(
        np.array(estimated_regime_seq),
        np.array(true_regime_seq),
    )
    estimated_regime_seq_with_aligned_labels = np.array(
        [perm_of_estimated[x] for x in estimated_regime_seq]
    )
    pct_correct_regimes = np.mean(true_regime_seq == estimated_regime_seq_with_aligned_labels)
    return pct_correct_regimes

def get_aligned_estimate(
    estimated_regime_seq,
    true_regime_seq,
) -> float:
    """
    Due to label-switching, need to find first find the permutation that best matches the truth
    in order to compute accuracy. This function returns the labels
    """
    # Convert types in case we have jax arrays with int32s.  This is necessary because
    # `find_permutation` does a type checking and assumes the type is int, not int32.
    estimated_regime_seq = np.asarray(estimated_regime_seq, dtype=int)
    true_regime_seq = np.asarray(true_regime_seq, dtype=int)

    # Rk: The `find_permutation` function requires numpy arrays
    perm_of_estimated = find_permutation(
        np.array(estimated_regime_seq),
        np.array(true_regime_seq),
    )
    estimated_regime_seq_with_aligned_labels = np.array(
        [perm_of_estimated[x] for x in estimated_regime_seq]
    )

    return estimated_regime_seq_with_aligned_labels