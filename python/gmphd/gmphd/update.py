from collections import namedtuple
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from copy import deepcopy

UpdateComponents = namedtuple('UpdateComponents',
                              ['meas_estimate', 'innovation_cov', 'kalman_gain', 'updated_cov'])


def compute_meas_estimate(gaussian, meas_model):
    return meas_model.dot(gaussian.mean)


def compute_innovation_cov(gaussian, meas_model, meas_noise):
    return meas_noise + meas_model.dot(gaussian.covariance.dot(meas_model.transpose()))


def compute_kalman_gain(gaussian, meas_model, innovation_cov):
    return gaussian.covariance.dot(meas_model.transpose().dot(inv(innovation_cov)))


def compute_updated_gauss_cov(gaussian, meas_model, kalman_gain):
    return gaussian.covariance - kalman_gain.dot(meas_model.dot(gaussian.covariance))


def compute_update_components(gaussian, meas_model, meas_noise):
    meas_estimate = compute_meas_estimate(gaussian, meas_model)
    innovation_cov = compute_innovation_cov(gaussian, meas_model, meas_noise)
    kalman_gain = compute_kalman_gain(gaussian, meas_model, innovation_cov)
    updated_cov = compute_updated_gauss_cov(gaussian, meas_model, kalman_gain)

    return UpdateComponents(meas_estimate, innovation_cov, kalman_gain, updated_cov)


def compute_updated_gaussian(gaussian, measurement, update_components, prob_detection):
    gaussian.weight = gaussian.weight * prob_detection * multivariate_normal.pdf(measurement,
                                                                                 mean=update_components.meas_estimate,
                                                                                 cov=update_components.innovation_cov)
    gaussian.mean = gaussian.mean + update_components.kalman_gain.dot(
        measurement - update_components.meas_estimate)
    gaussian.covariance = update_components.updated_cov

    return gaussian


def compute_non_detection_intensity(predicted_intensity, prob_detection):
    non_detection_intensity = list()
    for gaussian in predicted_intensity:
        updated_component = deepcopy(gaussian)
        updated_component.weight = updated_component.weight * (1 - prob_detection)
        non_detection_intensity.append(updated_component)

    return non_detection_intensity


def update_intensity(predicted_intensity, measurements, meas_model, meas_noise, prob_detection, clutter_model):
    updated_intensity = list()
    # For each predicted components, create a set of updated components handling the prob of non detection
    updated_intensity.extend(compute_non_detection_intensity(predicted_intensity, prob_detection))

    # Compute components required for update step (kalman gain, innovation_covariance, etc)
    update_components = list()
    for gaussian in predicted_intensity:
        update_components.append(compute_update_components(gaussian, meas_model, meas_noise))

    # For each measurement, generate a set of gaussian updated with the given measurement
    for measurement in measurements:
        measurement_updated_intensity = list()
        total_weight = 0.
        for gaussian, update_component in zip(predicted_intensity, update_components):
            updated_gaussian = deepcopy(gaussian)
            updated_gaussian = compute_updated_gaussian(updated_gaussian, measurement, update_component, prob_detection)
            total_weight = total_weight + updated_gaussian.weight
            measurement_updated_intensity.append(updated_gaussian)

        for gaussian in measurement_updated_intensity:
            gaussian.weight = gaussian.weight / (total_weight + clutter_model())

        updated_intensity.extend(measurement_updated_intensity)

    return updated_intensity
