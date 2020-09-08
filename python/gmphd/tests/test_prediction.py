from unittest import TestCase
from gmphd.prediction import predict_intensity
from gmphd.gaussian_component import GaussianComponent
from tests.models import TestBirthModel
import numpy as np


class TestPrediction(TestCase):
    def test_predict_single_component(self):
        dynamic_model = np.array([[1., 0., 1., 0.],
                                  [0., 1., 0., 1.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        process_noise = np.array([[0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05]])
        prob_survival = 0.95
        mean = np.array([1., 1., 0.5, 0.5]).transpose()
        covariance = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        weight = 1.
        gaussian_components = list()
        gaussian_components.append(GaussianComponent(mean, covariance, weight))

        predict_intensity(gaussian_components, dynamic_model, process_noise, prob_survival, TestBirthModel())

        self.assertTrue(np.allclose(gaussian_components[0].mean, [1.5, 1.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[0].covariance, np.array([[2.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 2.05, 0.05, 1.05],
                                                                                 [1.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 1.05, 0.05, 1.05]])))
        self.assertAlmostEqual(gaussian_components[0].weight, 0.95)

        self.assertTrue(np.allclose(gaussian_components[1].mean, [5., 5., 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[1].covariance, np.array([[1., 0., 0., 0.],
                                                                                 [0., 1., 0., 0.],
                                                                                 [0., 0., 1., 0.],
                                                                                 [0., 0., 0., 1.]])))
        self.assertAlmostEqual(gaussian_components[1].weight, 1.)

    def test_predict_multiple_components(self):
        dynamic_model = np.array([[1., 0., 1., 0.],
                                  [0., 1., 0., 1.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.]])
        process_noise = np.array([[0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05],
                                  [0.05, 0.05, 0.05, 0.05]])
        prob_survival = 0.95
        mean = np.array([1., 1., 0.5, 0.5]).transpose()
        covariance = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]])
        weight = 1.
        gaussian_components = list()
        gaussian_components.append(GaussianComponent(mean, covariance, weight))
        gaussian_components.append(GaussianComponent(mean, covariance, weight))

        predict_intensity(gaussian_components, dynamic_model, process_noise, prob_survival, TestBirthModel())

        self.assertTrue(np.allclose(gaussian_components[0].mean, [1.5, 1.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[0].covariance, np.array([[2.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 2.05, 0.05, 1.05],
                                                                                 [1.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 1.05, 0.05, 1.05]])))
        self.assertAlmostEqual(gaussian_components[0].weight, 0.95)
        self.assertTrue(np.allclose(gaussian_components[1].mean, [1.5, 1.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[1].covariance, np.array([[2.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 2.05, 0.05, 1.05],
                                                                                 [1.05, 0.05, 1.05, 0.05],
                                                                                 [0.05, 1.05, 0.05, 1.05]])))
        self.assertAlmostEqual(gaussian_components[1].weight, 0.95)
