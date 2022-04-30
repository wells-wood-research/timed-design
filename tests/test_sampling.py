import numpy as np

from utils.sampling_utils import random_choice_prob_index


def test_random_choice_prob_index():
    theoretical_prob = [
        [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.20,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.50,
            0.10,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.04,
        ],
    ]
    choices = [
        random_choice_prob_index(np.array(theoretical_prob), return_seq=False)[0]
        for i in range(1000000)
    ]
    # Count idxs vs all the choices:
    real_prob = np.bincount(choices) / float(len(choices))
    assert np.isclose(
        np.sum(real_prob), np.sum(theoretical_prob), rtol=0.01
    ), "Probabilities do not sum up to 1."
    # Test for recovered probability:
    assert np.allclose(
        np.array(theoretical_prob)[0], real_prob, rtol=0.01, atol=0.01
    ), f"Probabilities obtained are {np.array(theoretical_prob)[0]}, but got {real_prob}"
