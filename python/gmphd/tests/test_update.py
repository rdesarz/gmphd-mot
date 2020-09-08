from unittest import TestCase
import numpy as np

from gmphd.gaussian_component import GaussianComponent
from gmphd.update import *
import gmphd.clutter_models as clutter_model

# Parameters for the tests
gaussian_mean = np.array([1., 1., 0.5, 0.5]).transpose()
gaussian_cov = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
gaussian_weight = 1.
prob_detection = 0.5
measurement_model = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
measurement_noise = np.array([[1., 0.],
                              [0., 1.]])
measurement_1 = np.array([2., 2.]).transpose()
measurement_2 = np.array([1., 1.]).transpose()
measurements = [measurement_1, measurement_2]

clutter_model = clutter_model.Constant(1.)


class TestUpdate(TestCase):
    def test_non_detected_update_components(self):
        gaussian_components = list()
        gaussian_components.append(GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight))

        gaussian_components = compute_non_detection_intensity(gaussian_components, prob_detection)

        self.assertTrue(np.allclose(gaussian_components[0].mean, [1, 1, 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[0].covariance, np.array([[1., 0., 0., 0.],
                                                                                 [0., 1., 0., 0.],
                                                                                 [0., 0., 1., 0.],
                                                                                 [0., 0., 0., 1.]])))
        self.assertAlmostEqual(gaussian_components[0].weight, 1 - prob_detection)

    def test_compute_kalman_gain(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        innovation_covariance = compute_innovation_cov(component, measurement_model, measurement_noise)

        kalman_gain = compute_kalman_gain(component, measurement_model, innovation_covariance)

        self.assertTrue(np.allclose(kalman_gain, np.array([[0.5, 0.],
                                                           [0., 0.5],
                                                           [0., 0.],
                                                           [0., 0.]])))

    def test_compute_measurement_estimate(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        measurement_estimate = compute_meas_estimate(component, measurement_model)

        self.assertTrue(np.allclose(measurement_estimate, np.array([1., 1.]).transpose()))

    def test_compute_innovation_covariance(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        innovation_cov = compute_innovation_cov(component, measurement_model, measurement_noise)

        self.assertTrue(np.allclose(innovation_cov, np.array([[2., 0.],
                                                              [0., 2.]])))

    def test_compute_updated_covariance(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)
        kalman_gain = compute_kalman_gain(component, measurement_model, measurement_noise)

        updated_cov = compute_updated_gauss_cov(component, measurement_model, kalman_gain)

        self.assertTrue(np.allclose(updated_cov, np.array([[0., 0., 0., 0.],
                                                           [0., 0., 0., 0.],
                                                           [0., 0., 1., 0.],
                                                           [0., 0., 0., 1.]])))

    def test_compute_updated_gaussian(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)
        update_components = compute_update_components(component, measurement_model, measurement_noise)

        component = compute_updated_gaussian(component, measurement_1, update_components, prob_detection)

        self.assertTrue(np.allclose(component.mean, [1.5, 1.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(component.covariance, update_components.updated_cov))
        self.assertAlmostEqual(component.weight,
                               1. * prob_detection * multivariate_normal(update_components.meas_estimate,
                                                                         update_components.innovation_cov).pdf(
                                   measurement_1))

    def test_compute_updated_intensity_with_two_components(self):
        birth_component = GaussianComponent(np.array([2.5, 2.5, -0.1, 0.1]).transpose(),
                                            np.array([[0.5, 0., 0., 0.],
                                                      [0., 0.5, 0., 0.],
                                                      [0., 0., 0.1, 0.],
                                                      [0., 0., 0., 0.1]]),
                                            0.1)
        predicted_component_1 = GaussianComponent(np.array([-1.0, 2.5, 0.2, 0.1]).transpose(),
                                                  np.array([[0.5, 0., 0., 0.],
                                                            [0., 0.5, 0., 0.],
                                                            [0., 0., 0.1, 0.],
                                                            [0., 0., 0., 0.1]]),
                                                  0.8)
        predicted_component_2 = GaussianComponent(np.array([1.0, 2.5, 0.2, 0.1]).transpose(),
                                                  np.array([[0.5, 0., 0., 0.],
                                                            [0., 0.5, 0., 0.],
                                                            [0., 0., 0.1, 0.],
                                                            [0., 0., 0., 0.1]]),
                                                  0.8)
        intensity = list([deepcopy(birth_component), deepcopy(predicted_component_1), deepcopy(predicted_component_2)])

        measures = [np.array([-1.0, 2.5]),
                    np.array([1.2, 2.6]).transpose(),
                    np.array([2.0, 4.0]).transpose()]

        intensity = update_intensity(intensity, measures, measurement_model, measurement_noise, prob_detection, clutter_model)

        # Compute updated with first measurement for test purpose
        weight_1 = multivariate_normal.pdf(measures[0],
                                           mean=np.array([2.5, 2.5]),
                                           cov=np.array([[1.5, 0.],
                                                         [0., 1.5]]))
        weight_2 = multivariate_normal.pdf(measures[0],
                                           mean=np.array([-1.0, 2.5]),
                                           cov=np.array([[1.5, 0.],
                                                         [0., 1.5]]))
        weight_3 = multivariate_normal.pdf(measures[0],
                                           mean=np.array([1.0, 2.5]),
                                           cov=np.array([[1.5, 0.],
                                                         [0., 1.5]]))

        weight_1 = weight_1 * 0.1 * prob_detection
        weight_2 = weight_2 * 0.8 * prob_detection
        weight_3 = weight_3 * 0.8 * prob_detection

        total_weight = weight_1 + weight_2 + weight_3

        weight_1 = weight_1 / (total_weight + clutter_model())
        weight_2 = weight_2 / (total_weight + clutter_model())
        weight_3 = weight_3 / (total_weight + clutter_model())

        self.assertTrue(np.allclose(intensity[0].mean, birth_component.mean))
        self.assertTrue(np.allclose(intensity[1].mean, predicted_component_1.mean))
        self.assertTrue(np.allclose(intensity[2].mean, predicted_component_2.mean))
        self.assertAlmostEqual(intensity[0].weight, birth_component.weight * (1 - prob_detection))
        self.assertAlmostEqual(intensity[1].weight, predicted_component_1.weight * (1 - prob_detection))
        self.assertAlmostEqual(intensity[2].weight, predicted_component_2.weight * (1 - prob_detection))
        self.assertAlmostEqual(intensity[3].weight, weight_1)
        self.assertAlmostEqual(intensity[4].weight, weight_2)
        self.assertAlmostEqual(intensity[5].weight, weight_3)
