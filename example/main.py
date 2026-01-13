from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataProvider import DataProvider
from pipelineBuilder import PipelineBuilder
from experimentRunner import ExperimentRunner
from pathlib import Path
import pandas as pd

#Prepare the Data
DATA_DIR = Path(__file__).resolve().parent / "data"
df = pd.read_csv(DATA_DIR / "enron_spam_data.csv")
df = df.dropna(subset=["Message"]).reset_index(drop=True)

# Down sample
df_ds = df.sample(frac=0.5, random_state=42)

df_ds["Spam/Ham"] = df_ds["Spam/Ham"].map({
    "ham": 0,
    "spam": 1
})

X = df_ds["Message"]
y = df_ds["Spam/Ham"]

# CV and Scoring
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall"
}

data_provider = DataProvider(
    X=X,
    y=y,
    mode="cv",          # or "holdout"
    stratify=True
)


# Linear SVM

linear_svm_pipeline = PipelineBuilder(
    transformers=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        ))
    ],
    estimator=("clf", LinearSVC(dual=False, max_iter=5000))
).build()


linear_svm_runner = ExperimentRunner(
    data_provider=data_provider,
    pipeline=linear_svm_pipeline,
    scoring=scoring,
    cv_strategy=cv
)

linear_svm_results = linear_svm_runner.run()


# RBF SVM

rbf_svm_pipeline = PipelineBuilder(
    transformers=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=5000   # IMPORTANT for RBF
        ))
    ],
    estimator=("clf", SVC(kernel="rbf"))
).build()


rbf_svm_runner = ExperimentRunner(
    data_provider=data_provider,
    pipeline=rbf_svm_pipeline,
    scoring=scoring,
    cv_strategy=cv,
    # param_grid={
    #     # TfidfVectorizer hyperparameters
    #     "tfidf__ngram_range": [(1, 1), (1, 2)],
    #     "tfidf__min_df": [1, 2],
    #     "tfidf__max_df": [0.9, 0.95],
    #     "tfidf__max_features": [3000, 5000],
    #     # RBF SVM hyperparameters
    #     "clf__C": [0.1, 1, 10],
    #     "clf__gamma": ["scale", 0.01, 0.001]
    # },
    # refit_metric="f1",
    verbose=1
)

rbf_svm_results = rbf_svm_runner.run()


# Logistic Regression

logreg_pipeline = PipelineBuilder(
    transformers=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        ))
    ],
    estimator=("clf", LogisticRegression(
        solver="liblinear",
        max_iter=5000
    ))
).build()


logreg_runner = ExperimentRunner(
    data_provider=data_provider,
    pipeline=logreg_pipeline,
    scoring=scoring,
    cv_strategy=cv
)

logreg_results = logreg_runner.run()

# Decision Tree

dt_pipeline = PipelineBuilder(
    transformers=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=3000   # Trees need tight control
        ))
    ],
    estimator=("clf", DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        random_state=42
    ))
).build()


dt_runner = ExperimentRunner(
    data_provider=data_provider,
    pipeline=dt_pipeline,
    scoring=scoring,
    cv_strategy=cv
)

dt_results = dt_runner.run()


# Compare results of all models

results = {
    "Linear SVM": linear_svm_results,
    "RBF SVM": rbf_svm_results,
    "Logistic Regression": logreg_results,
    "Decision Tree": dt_results
}

for model, res in results.items():
    print(f"\n{model}")
    # Handle grid search vs plain CV results
    if res.get("search_type") == "grid":
        print("Best CV score:", res["best_score"])
        print("Best params:", res["best_params"])
        print("Total time (s):", res["total_time"])
    else:
        for metric, mean in res["scores"].items():
            if metric.startswith("test_"):
                print(metric, mean.mean())
        print("Total time (s):", res["total_time"])
