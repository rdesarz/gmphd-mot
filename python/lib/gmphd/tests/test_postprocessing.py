import numpy as np
from unittest import TestCase
from gmphd.postprocessing import *


gaussian_mean = np.array([1., 1., 0.5, 0.5]).transpose()
gaussian_cov = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
gaussian_weight = 1.
truncation_threshold = 0.1


class TestPostprocessing(TestCase):
    def test_pruning(self):
        posterior_intensity = list()
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight))
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, truncation_threshold))
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, 0.01))

        posterior_intensity = apply_pruning(posterior_intensity, truncation_threshold)

        self.assertEqual(len(posterior_intensity), 2)
        self.assertAlmostEqual(posterior_intensity[0].weight, 1.)
        self.assertAlmostEqual(posterior_intensity[1].weight, truncation_threshold)

    def test_merge_score_same_gaussian(self):
        gaussian = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        score = compute_merge_score(gaussian, gaussian)

        self.assertAlmostEqual(score, 0.)

    def test_merge_score_different_gaussian(self):
        gaussian_1 = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)
        gaussian_mean_2 = np.array([0.5, 0.5, 0.2, 0.2]).transpose()
        gaussian_cov_2 = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
        gaussian_weight_2 = 1.
        gaussian_2 = GaussianComponent(gaussian_mean_2, gaussian_cov_2, gaussian_weight_2)

        score = compute_merge_score(gaussian_1, gaussian_2)

        self.assertAlmostEqual(score, 0.68)

    def test_merge_three_equal_gaussians(self):
        gaussian_1 = GaussianComponent(gaussian_mean, gaussian_cov, 0.1)
        gaussian_2 = GaussianComponent(gaussian_mean, gaussian_cov, 0.1)
        gaussian_3 = GaussianComponent(gaussian_mean, gaussian_cov, 0.1)

        merged_gaussian = merge_gaussians([gaussian_1, gaussian_2, gaussian_3])

        self.assertAlmostEqual(merged_gaussian.weight, 0.3)
        self.assertTrue(np.allclose(merged_gaussian.mean, gaussian_mean))
        self.assertTrue(np.allclose(merged_gaussian.covariance, gaussian_cov))

    def test_merge_intensity_with_three_gaussian_two_close(self):
        gaussian_1 = GaussianComponent(gaussian_mean, gaussian_cov, 0.6)
        gaussian_2 = GaussianComponent(gaussian_mean, gaussian_cov, 0.3)
        gaussian_mean_2 = np.array([0.5, 0.5, 0.2, 0.2]).transpose()
        gaussian_cov_2 = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
        gaussian_3 = GaussianComponent(gaussian_mean_2, gaussian_cov_2, 0.1)

        posterior_intensity = [gaussian_1, gaussian_2, gaussian_3]

        posterior_intensity = apply_merging(posterior_intensity, 0.5)

        self.assertEqual(len(posterior_intensity), 2)

    def test_merge_intensity_with_three_gaussian_two_close_not_weight_sorted(self):
        gaussian_1 = GaussianComponent(gaussian_mean, gaussian_cov, 0.6)
        gaussian_2 = GaussianComponent(gaussian_mean, gaussian_cov, 0.3)
        gaussian_mean_2 = np.array([0.5, 0.5, 0.2, 0.2]).transpose()
        gaussian_cov_2 = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
        gaussian_3 = GaussianComponent(gaussian_mean_2, gaussian_cov_2, 0.1)

        posterior_intensity = [gaussian_2, gaussian_3, gaussian_1]

        posterior_intensity = apply_merging(posterior_intensity, 0.5)

        self.assertEqual(len(posterior_intensity), 2)
        self.assertAlmostEqual(posterior_intensity[0].weight, 0.9)

    def test_merge_intensity_with_five_gaussian_two_close_not_weight_sorted(self):
        gaussian_1 = GaussianComponent(gaussian_mean, gaussian_cov, 0.6)
        gaussian_2 = GaussianComponent(gaussian_mean, gaussian_cov, 0.3)
        gaussian_mean_2 = np.array([0.5, 0.5, 0.2, 0.2]).transpose()
        gaussian_cov_2 = np.array([[1., 0., 0., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])
        gaussian_3 = GaussianComponent(gaussian_mean_2, gaussian_cov_2, 0.1)
        gaussian_4 = GaussianComponent(gaussian_mean_2, gaussian_cov_2, 0.1)

        posterior_intensity = [gaussian_2, gaussian_3, gaussian_1, gaussian_4]

        posterior_intensity = apply_merging(posterior_intensity, 0.5)

        self.assertEqual(len(posterior_intensity), 2)

    def test_merge_empty_intensity(self):
        posterior_intensity = []

        posterior_intensity = apply_merging(posterior_intensity, 0.5)

        self.assertEqual(len(posterior_intensity), 0)
