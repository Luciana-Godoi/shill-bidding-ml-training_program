# shill-bidding-ml-training_program

# Machine learning and Bayesian modeling for imbalanced classification problems, focusing on interpretability and threshold optimization using the Shill Bidding dataset (UFES Training Program).


## Overview

This repository contains the full implementation, analysis, and results of a comparative study between **Bayesian modeling** and **machine learning algorithms** applied to the **Shill Bidding dataset**, a benchmark dataset for fraud detection in online auctions.

The project was developed as part of the **Licença para Capacitação program at the Universidade Federal do Espírito Santo (UFES)**, under the training theme:

> *Machine Learning: Exploring Techniques, Strategies for Imbalanced Data, and Model Interpretability.*

The main objective of this training was to strengthen theoretical and practical knowledge in machine learning, with emphasis on imbalanced data problems and model interpretability.

---

## Objectives

This project addresses three main research axes:

### 1. Machine Learning Methods

Implementation and comparison of several supervised learning models, including:

* Logistic Regression (LASSO)
* Linear, Polynomial, and RBF Support Vector Machines
* Decision Trees
* Bagging
* Random Forest
* K-Nearest Neighbors (KNN)
* Linear and Quadratic Discriminant Analysis (LDA/QDA)
* Gradient Boosting (XGBoost)

---

### 2. Bayesian Modeling

A Bayesian Power Cauchy model, originally implemented in **PyStan 2** by Alex de la Cruz Huayanay, Jorge L. Bazán, and Cibele M. Russo (see *“Performance of Evaluation Metrics for Classification in Imbalanced Data”*, https://doi.org/10.1007/s00180-024-01539-5), was adapted to **PyStan 3.10.1**, ensuring full reproducibility of the published results under the updated API.

In addition to the original implementation, this work introduces the following extensions:

* Full covariate analysis using the original feature set;
* Estimation of Highest Posterior Density (HPD) intervals;
* MCMC convergence diagnostics, including traceplots and autocorrelation analysis;
* Comparison of alternative variable selection strategies under the Bayesian framework.

---

### 3. Imbalanced Data and Threshold Optimization

Given the highly imbalanced nature of the dataset, different strategies were evaluated:

* Oversampling and class balancing approaches
* Cross-validation schemes (10-fold × 3 repeats)
* Threshold selection strategies:

  * Based on training set probabilities
  * Based on out-of-fold cross-validation predictions

---

### 4. Model Interpretability

Model explainability was investigated using:

* SHAP (Shapley Additive Explanations)
* Feature importance analysis
* Comparison of variable relevance across models
* Bayesian posterior interpretation

---

## Key Contributions

* Reimplementation of a Bayesian model in PyStan 3.10.1
* Reproduction of published results (Power Cauchy model)
* Extension of Bayesian analysis using full covariate space
* Identification of additional predictive variables using XGBoost (e.g., `Bidding_Ratio`)
* Comparative evaluation of multiple machine learning models
* Analysis of two different threshold selection strategies
* Integration of explainability techniques (SHAP) across models

---

## Repository Structure

```
python/
    Bayesian models (PyStan implementations)

R/
    Machine learning models, XGBoost, evaluation pipelines

data/
    Raw and processed datasets

figures/
    SHAP plots, MCMC diagnostics, comparison charts

results/
    Performance metrics and model outputs

docs/
    Methodological details
```


## Training Context (UFES)

This work was developed as part of the **Licença para Capacitação at UFES**, under the project:

> *Machine Learning: Exploring Techniques, Strategies for Imbalanced Data, and Model Interpretability.*

It reflects the applied component of the training program, integrating theoretical study and computational implementation.

---

## References

Key references include:

* de la Cruz Huayanay, Bazán & Russo (2025)
* Hastie, Tibshirani & Friedman (2009)
* James et al. (2013, 2023)
* Murphy (2012)
* Bishop (1995)
* Chawla et al. (2002)
* Lundberg & Lee (2017)
* He & Garcia (2009)
* Gelman et al. (2013)

---


