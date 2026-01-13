from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import get_scorer
import time

class ExperimentRunner:
    def __init__(
        self,
        data_provider,
        pipeline,
        scoring,
        cv_strategy=None,
        param_grid=None,
        refit_metric=None,
        n_jobs=-1,
        verbose=0
    ):
        """
        Parameters
        ----------
        data_provider : DataProvider
            Provides data (CV or holdout mode)
        pipeline : sklearn Pipeline
            ML pipeline to evaluate
        scoring : str or callable
            sklearn-compatible scoring
        cv_strategy : CV splitter (optional)
            StratifiedKFold, KFold, etc.
        param_grid : dict (optional)
            Enables GridSearchCV if provided
        """
        self.data_provider = data_provider
        self.pipeline = pipeline
        self.scoring = scoring
        self.cv_strategy = cv_strategy
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit_metric = refit_metric

        self._validate_configuration()
        self.mean_scores = []
        self.std_scores = []

    def _validate_configuration(self):
        if self.data_provider.mode == "cv" and self.cv_strategy is None:
            raise ValueError("CV strategy must be provided when mode='cv'")

    def run(self):
        if self.data_provider.mode == "cv":
            return self._run_cv_experiment()
        else:
            return self._run_holdout_experiment()

    def _run_cv_experiment(self):
        X, y = self.data_provider.get_data()

        start_time = time.time()

        if self.param_grid:
            search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                scoring=self.scoring,
                refit=self.refit_metric,
                cv=self.cv_strategy,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )

            search.fit(X, y)

            elapsed_time = time.time() - start_time

            return {
                "mode": "cv",
                "search_type": "grid",
                "best_score": search.best_score_,
                "best_params": search.best_params_,
                "cv_results": search.cv_results_,
                "total_time": elapsed_time
            }

        else:
            scores = cross_validate(
                estimator=self.pipeline,
                X=X,
                y=y,
                cv=self.cv_strategy,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                return_train_score=True
            )

            elapsed_time = time.time() - start_time

            for metric in self.scoring.keys():
              test_scores = scores[f"test_{metric}"]
              self.mean_scores.append((metric, test_scores.mean()))
              self.std_scores.append((metric, test_scores.std()))

            return {
                "mode": "cv",
                "search_type": "none",
                "scores": scores,
                "total_time": elapsed_time
            }

    def _run_holdout_experiment(self):
        X_train, X_test, y_train, y_test = self.data_provider.get_train_test_data()

        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time

        start_time = time.time()
        score = get_scorer(self.scoring)(self.pipeline, X_test, y_test)
        eval_time = time.time() - start_time

        return {
            "mode": "holdout",
            "score": score,
            "fit_time": fit_time,
            "eval_time": eval_time
        }
