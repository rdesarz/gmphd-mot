from numpy.linalg import inv
from numpy import float64
from math import fsum
from gmphd.gaussian_component import GaussianComponent


def apply_pruning(posterior_intensity, truncation_threshold):
    posterior_intensity = [gaussian for gaussian in posterior_intensity if gaussian.weight >= truncation_threshold]
    return posterior_intensity


def apply_merging(posterior_intensity, merge_threshold):
    # The merging algorithm process the components in sorted order based on their weight
    posterior_intensity.sort(key=lambda gaussian: gaussian.weight, reverse=True)

    # Lookup table to check if a component was merged
    merged = [False for i in range(len(posterior_intensity))]

    # Process the gaussian components in a pyramidal fashion
    for i in range(0, len(posterior_intensity) - 1):
        if not merged[i]:
            to_be_merged = [posterior_intensity[i]]
            for j in range(i + 1, len(posterior_intensity)):
                if not merged[j] and compute_merge_score(posterior_intensity[i],
                                                         posterior_intensity[j]) < merge_threshold:
                    to_be_merged.append(posterior_intensity[j])
                    merged[j] = True

                posterior_intensity[i] = merge_gaussians(to_be_merged)

    posterior_intensity = [component for merged, component in zip(merged, posterior_intensity) if not merged]

    return posterior_intensity


def merge_gaussians(gaussians_list):
    weight = fsum(gaussian.weight for gaussian in gaussians_list)
    mean = sum(gaussian.mean.dot(gaussian.weight) for gaussian in gaussians_list).dot(1. / weight)
    covariance = sum(
        (gaussian.covariance + (mean - gaussian.mean).transpose().dot((mean - gaussian.mean))).dot(gaussian.weight) for
        gaussian in gaussians_list).dot(
        float64(1.) / weight)

    return GaussianComponent(weight=weight, mean=mean, covariance=covariance)


def compute_merge_score(gaussian_i, gaussian_j):
    mean_diff = (gaussian_i.mean - gaussian_j.mean)
    return mean_diff.transpose().dot(inv(gaussian_i.covariance).dot(mean_diff))
