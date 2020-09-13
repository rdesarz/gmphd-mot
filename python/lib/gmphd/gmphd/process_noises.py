import numpy as np


class ConstantVelocity:
    def __init__(self, process_noise_std_dev):
        self.process_noise_std_dev = process_noise_std_dev

    def __call__(self, delta_t):
        return np.array([[pow(delta_t, 4) / 4., 0., pow(delta_t, 3) / 2, 0.],
                         [0., pow(delta_t, 4) / 4, 0., pow(delta_t, 3) / 2],
                         [pow(delta_t, 3) / 2., 0., pow(delta_t, 2), 0.],
                         [0., pow(delta_t, 3) / 2, 0., pow(delta_t, 2)]], dtype=np.float64).dot(
            pow(self.process_noise_std_dev, 2))
