from sklearn.pipeline import Pipeline

class PipelineBuilder:
    def __init__(self, transformers=None, estimator=None):
        """
        Parameters
        ----------
        transformers : list of (str, transformer)
            Feature extraction / preprocessing steps
        estimator : (str, estimator)
            Final model step
        """
        self.transformers = transformers or []
        self.estimator = estimator

        self._validate_inputs()

    def _validate_inputs(self):
        if not self.estimator:
            raise ValueError("An estimator must be provided")

        for name, transformer in self.transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError(
                    f"Transformer '{name}' must implement fit and transform"
                )

        est_name, est = self.estimator
        if not hasattr(est, "fit") or not hasattr(est, "predict"):
            raise TypeError(
                f"Estimator '{est_name}' must implement fit and predict"
            )

    def build(self):
        steps = []

        for name, transformer in self.transformers:
            steps.append((name, transformer))

        steps.append(self.estimator)

        return Pipeline(steps)
