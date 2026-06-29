# Impact of Label Noise on Synthetic Tabular Data Generation using ACTG175

This repository contains the code developed to study the impact of label noise on synthetic tabular data generation using the ACTG175 dataset.

## Project Goal

The main goal of this project is to evaluate how increasing levels of label noise affect the generation and usefulness of synthetic tabular data. The study compares three generators:

* Gaussian Copula Synthesizer (GCS)
* CTGAN
* TVAE

The experiments are conducted on the ACTG175 dataset and are evaluated from three main perspectives:

* statistical quality
* downstream classification utility
* fairness

## Dataset

The ACTG175 dataset is a healthcare-related tabular dataset derived from clinical trial data. In this project, the target variable is `cid`, treated as a binary classification target.

The dataset is used to analyze how label noise affects:

* the statistical fidelity of synthetic data
* its usefulness for classification tasks
* performance differences across relevant groups, especially gender and race

## Repository Structure

The main source code is organized in the `src/` folder.

### Source files

* `prepare_dataset.py` – prepares and cleans the ACTG175 dataset
* `experiment_data.py` – prepares the train/test split and dataset organization
* `data_utils.py` – utility functions for loading and handling data
* `synthetic_utils.py` – helper functions related to synthetic data generation
* `evaluation_utils.py` – shared utilities for evaluation
* `classification_utils.py` – helper functions for downstream classification
* `fairness_utils.py` – helper functions for fairness evaluation
* `statistical_metrics.py` – utilities for computing statistical metrics

### Experiment scripts

* `run_experiments.py` – runs the synthetic data generation experiments
* `run_statistical_metrics.py` – computes the statistical quality metrics
* `run_classification.py` – runs downstream classification experiments
* `run_fairness.py` – runs fairness analysis

### Plot scripts

* `plot_statistical_metrics.py` – generates statistical quality plots
* `plot_classification_results.py` – generates classification plots
* `plot_fairness_results.py` – generates fairness gap plots
* `plot_fairness_group_results.py` – generates detailed fairness-by-group plots

## Output Files

The pipeline produces:

* clean and processed datasets in CSV format
* noisy versions of the training data
* synthetic datasets generated with GCS, CTGAN, and TVAE
* statistical result files
* classification result files
* fairness result files
* plots in PNG format for:

  * statistical metrics
  * classification performance
  * fairness gaps
  * fairness by group

## Execution Order

Execution flow:

1. prepare the dataset
2. create the train/test split
3. run synthetic data generation experiments
4. compute statistical metrics
5. run classification experiments
6. run fairness experiments
7. generate plots

Order of scripts:

1. `src/prepare_dataset.py`
2. `src/experiment_data.py`
3. `src/run_experiments.py`
4. `src/run_statistical_metrics.py`
5. `src/run_classification.py`
6. `src/run_fairness.py`
7. `src/plot_statistical_metrics.py`
8. `src/plot_classification_results.py`
9. `src/plot_fairness_results.py`
10. `src/plot_fairness_group_results.py`

## Main Libraries Used

This project was developed in Python and mainly uses:

* `pandas`
* `matplotlib`
* `scikit-learn`
* `sdv`
* `pyMDMA`

## Notes

* The experiments focus on increasing levels of label noise applied to the target variable.
* The target variable used in this project is `cid`.
* The evaluation is performed through statistical quality, downstream predictive utility, and fairness.
* The fairness analysis is mainly based on group attributes such as gender and race.

## Author

Martim Fernandes
