# HeartPredict <!-- omit in toc -->

![logo](/docs/logo/logo.png)

HeartPredict is a Python library designed to analyze
and predict heart failure outcomes using patient data.

- [Dataset information](#dataset-information)
- [Key Questions to Answer with the Dataset](#key-questions-to-answer-with-the-dataset)
  - [Descriptive Analysis](#descriptive-analysis)
  - [Correlation and Feature Importance](#correlation-and-feature-importance)
  - [Predictive Analysis](#predictive-analysis)
  - [Survival Analysis](#survival-analysis)
  - [Risk Factor Analysis](#risk-factor-analysis)
- [Usage](#usage)
  - [Installation](#installation)
  - [CLI](#cli)
  - [Docker](#docker)
  - [Notebook](#notebook)
- [Contributing](#contributing)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

## Dataset information

The dataset used for this analysis was obtained from kaggle.com.
It contains 5000 medical records of patients who had heart-failure
and is licensed under CC0; made available under [this URL](https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records).

## Key Questions to Answer with the Dataset

### Descriptive Analysis

- What are the basic statistics (mean, median, standard deviation)
  of the clinical features?
- How is the age distribution of patients?
- What is the proportion of patients with conditions like anaemia, diabetes
  and high blood pressure?

### Correlation and Feature Importance

- Which clinical features are most strongly correlated with the DEATH_EVENT?
  And what are the most important features for predicting heart failure outcomes?
- How do different clinical features contribute
  to the risk of death due to heart failure?

### Predictive Analysis

- How accurately can we predict DEATH_EVENT using clinical features?
- Which machine learning model performs best for this prediction task?

### Survival Analysis

- Can we identify patient subgroups with higher or lower survival probabilities? 
- And what is the survival rate of patients over the follow-up period?

### Risk Factor Analysis

- How does smoking affect the risk of death in heart failure patients?
- What is the impact of serum creatinine and serum sodium levels on patient outcomes?
- How does the combination of multiple risk factors affect the likelihood
  of heart failure-related death?

## Usage

### Installation

You may want to use a virtual environment to install into.

```bash
pip install git+https://github.com/HeartPredict/HeartPredict
```

### CLI

Once installed, the `hp` CLI app should be available
in your virtual environment or system.
You can simply run `hp` to get a list of available options and commands.

### Docker

You can also use the CLI via docker
by cloning the repository
and running the following command:

```bash
docker build -t hp --rm . && docker run -it --name hp --rm hp
```

When you're done, simply exit the container with `exit`.

### Notebook

We also provide you with an interactive Jupyter Notebook
that visualizes our results.
It can be found [here](./notebook/heart_predict.ipynb).

## Contributing

We welcome contributions from the community!
If you're interested in contributing to HeartPredict,
please take a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
It contains all the guidelines you need to follow to get started,
including how to report issues, suggest features, and submit code.

## Code of Conduct

We are committed to providing a friendly, safe
and welcoming environment for everyone.
Please read our [Code of Conduct](CONDUCT.md)
to understand the standards we expect all members of our community to adhere to.

## License

This project is licensed under the MIT License -
see the [LICENSE](LICENSE) file for details.
