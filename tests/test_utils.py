import numpy as np

from design_utils.utils import compress_rotamer_predictions_to_20


def test_compress_rotamer_predictions_to_20():
    prediction_matrix = np.ones((1, 338))
    reduced_matrix = compress_rotamer_predictions_to_20(prediction_matrix)
    assert (
        reduced_matrix.shape[-1] == 20
    ), "Expected shape of compressed rotamers is 20."
