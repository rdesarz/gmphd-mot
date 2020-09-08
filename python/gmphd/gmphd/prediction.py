from copy import deepcopy


def predict_intensity(intensity, dynamic_model, process_noise, prob_survival, birth_intensity):
    for gaussian in intensity:
        gaussian.mean = dynamic_model.dot(gaussian.mean)
        gaussian.covariance = dynamic_model.dot(gaussian.covariance).dot(dynamic_model.transpose()) + process_noise
        gaussian.weight = gaussian.weight * prob_survival

    intensity.extend(deepcopy(birth_intensity()))

    return intensity
