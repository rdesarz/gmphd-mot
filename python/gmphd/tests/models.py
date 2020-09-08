from gmphd.gaussian_component import GaussianComponent
import numpy as np


class TestBirthModel:
    def __call__(self):
        birth_intensity = list()
        mean = np.array([5., 5., 0.5, 0.5]).transpose()
        covariance = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        weight = 1.
        birth_intensity.append(GaussianComponent(mean=mean, covariance=covariance, weight=weight))
        return birth_intensity
