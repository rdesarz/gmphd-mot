import numpy as np


class ConstantVelocity:
    def __call__(self, delta_t):
        return np.array([[1, 0., delta_t, 0.],
                         [0., 1., 0., delta_t],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]], dtype=np.float64)
