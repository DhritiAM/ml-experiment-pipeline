from sklearn.model_selection import train_test_split

class DataProvider:
    def __init__(
        self,
        X,
        y,
        mode="cv",
        test_size=0.2,
        random_state=42,
        stratify=True
    ):
        """
        Parameters
        ----------
        X : array-like
            Features (raw or preprocessed)
        y : array-like
            Labels
        mode : str
            'cv' or 'holdout'
        test_size : float
            Used only in holdout mode
        random_state : int
            Reproducibility
        stratify : bool
            Whether to stratify split (classification tasks)
        """
        self.X = X
        self.y = y
        self.mode = mode
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        self._validate_mode()

    def _validate_mode(self):
        if self.mode not in ["cv", "holdout"]:
            raise ValueError("mode must be either 'cv' or 'holdout'")

    def get_data(self):
        """
        Used when cross-validation is enabled.
        Returns full dataset.
        """
        if self.mode != "cv":
            raise RuntimeError("get_data() is only valid in CV mode")
        return self.X, self.y

    def get_train_test_data(self):
        """
        Used when cross-validation is disabled.
        Returns train-test split.
        """
        if self.mode != "holdout":
            raise RuntimeError("get_train_test_data() is only valid in holdout mode")

        stratify_labels = self.y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_labels
        )

        return X_train, X_test, y_train, y_test