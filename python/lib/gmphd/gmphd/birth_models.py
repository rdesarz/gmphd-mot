import numpy as np
from numpy import float64
from gmphd.gaussian_component import GaussianComponent


# Birth model which expects target to appear on the two lateral sides of a square field of view
class SquaredFieldOfView:
    def __init__(self, width, depth, step=1.0):
        self.width = width
        self.depth = depth
        self.step = step

    def __call__(self):
        birth_intensity = list()

        # Birth intensity on right side of FOV
        for depth in np.arange(0., self.depth, self.step):
            mean = np.array([self.width / 2., depth, -0.2, 0.], dtype=float64).transpose()
            covariance = np.array([[self.step, 0., 0., 0.],
                                   [0., 0.5, 0., 0.],
                                   [0., 0., 0.1, 0.],
                                   [0., 0., 0., 0.1]], dtype=float64)
            weight = float64(0.1)
            birth_intensity.append(GaussianComponent(mean=mean, covariance=covariance, weight=weight))

        # Birth intensity on left side of FOV
        for depth in np.arange(0., self.depth, self.step):
            mean = np.array([-self.width / 2., depth, 0.1, 0.], dtype=float64).transpose()
            covariance = np.array([[self.step, 0., 0., 0.],
                                   [0., 0.5, 0., 0.],
                                   [0., 0., 0.1, 0.],
                                   [0., 0., 0., 0.1]], dtype=float64)
            weight = float64(0.1)
            birth_intensity.append(GaussianComponent(mean=mean, covariance=covariance, weight=weight))

        return birth_intensity
