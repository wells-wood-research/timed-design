import numpy as np

from utils.sampling_utils import apply_temp_to_probs, random_choice_prob_index

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


def test_random_choice_prob_index():
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


def test_apply_temp_to_probs():
    new_probs = apply_temp_to_probs(probs=theoretical_prob, t=1)
    assert np.allclose(
        new_probs, theoretical_prob
    ), "Temperature Factor of 1 changed probabilities"
    new_probs = apply_temp_to_probs(probs=theoretical_prob, t=0.01)
    assert np.argmax(new_probs) == np.argmax(
        theoretical_prob
    ), "Argmax has changed with lower temperature"
    assert np.isclose(
        new_probs[:, np.argmax(new_probs)], 1.0
    ), "Argmax should have probability close to 1 at t=0.01"
    new_probs = apply_temp_to_probs(probs=theoretical_prob, t=100)
    assert np.allclose(
        np.array([1 / 20] * 20), new_probs, rtol=0.01, atol=0.01
    ), f"At higher temperature factors, probabilities shoudld be close to 0.05 but got {new_probs[0]}"
