from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.linear_model import Ridge

from models.mlp.train import FactorMLP


class AdditionalModelTests(unittest.TestCase):
    def test_ridge_can_fit_factor_matrix(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(200, 25)).astype("float32")
        y = (x[:, 0] * 0.2 - x[:, 1] * 0.1).astype("float32")
        model = Ridge(alpha=10.0, solver="lsqr").fit(x, y)
        self.assertEqual(model.predict(x[:7]).shape, (7,))

    def test_mlp_output_shape(self):
        model = FactorMLP(25)
        output = model(torch.randn(32, 25))
        self.assertEqual(tuple(output.shape), (32,))


if __name__ == "__main__":
    unittest.main()
