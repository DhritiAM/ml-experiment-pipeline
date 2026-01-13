# ML Experiment Pipeline

A generic, sklearn-compatible machine learning experimentation framework designed for rapid prototyping and systematic model evaluation.

## Overview

This framework provides a unified interface for ML experimentation with:
- **Dataset-agnostic data providers** for seamless data loading
- **Dynamic pipeline construction** supporting arbitrary sklearn transformers and estimators
- **Centralized experiment orchestration** with reproducible configurations
- **Flexible evaluation strategies** including cross-validation, holdout sets, and hyperparameter tuning

## Key Features

- Sklearn-compatible API for drop-in integration
- Support for custom preprocessing pipelines
- Multiple evaluation strategies (cross-validation, train-test split)
- Integrated hyperparameter tuning via GridSearchCV
- Experiment tracking and result logging

## Example Use Cases

- Compare multiple preprocessing strategies across different models
- Systematically evaluate feature engineering approaches
- Benchmark model performance across multiple datasets
- Reproduce experimental results with consistent configurations

## Requirements

- Python 3.8+
- scikit-learn >= 1.0
- numpy
- pandas

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
