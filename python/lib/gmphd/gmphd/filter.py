from gmphd.update import update_intensity
from gmphd.prediction import predict_intensity
from gmphd.postprocessing import apply_pruning, apply_merging


class GaussianMixturePhdFilter:
    def __init__(self, dynamic_model, process_noise, prob_survival, birth_intensity,
                 measurement_model, measurement_noise, prob_detection, clutter_model, pruning_threshold,
                 merging_threshold, extraction_threshold):
        self.dynamic_model = dynamic_model
        self.process_noise = process_noise
        self.prob_survival = prob_survival
        self.birth_intensity = birth_intensity
        self.measurement_model = measurement_model
        self.measurement_noise = measurement_noise
        self.prob_detection = prob_detection
        self.clutter_model = clutter_model
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.extraction_threshold = extraction_threshold
        self.current_intensity = list()

    def predict(self, delta_t):
        predict_intensity(self.current_intensity, self.dynamic_model(delta_t), self.process_noise(delta_t),
                          self.prob_survival, self.birth_intensity)

    def update(self, measurements):
        update_intensity(self.current_intensity, measurements, self.measurement_model, self.measurement_noise,
                         self.prob_detection, self.clutter_model)
        apply_pruning(self.current_intensity, self.pruning_threshold)
        apply_merging(self.current_intensity, self.merging_threshold)

    def get_intensity(self):
        return self.current_intensity

    def get_extracted_targets(self):
        return [component for component in self.current_intensity if component.weight > self.extraction_threshold]
