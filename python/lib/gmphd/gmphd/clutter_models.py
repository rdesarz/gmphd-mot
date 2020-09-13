class Constant:
    def __init__(self, intensity):
        self.intensity = intensity

    def __call__(self):
        return self.intensity
