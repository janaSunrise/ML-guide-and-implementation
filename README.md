# Sklearn dataset predictions

This repository contains the predictions, and plots 
for the datasets included in the scikit learn library 
by default and also some other datasets from kaggle or other sources.

## Datasets implemented

### Diabetes: 

This dataset consists of 9 columns namely
- pregnancies
- glucose
- diastolic
- triceps
- insulin
- bmi
- dpf
- age
- diabetes

The target value which has to be predicted is `diabetes`

This is a classifier problem, where the value of diabetes in boolean,
but in integer format.

Algorithm used for the problem: `GradientBoostingClassifier`

Accuracy achieved: `0.74`

### Housing

### Wine

## 🛠️ Tech stack used

- `Pandas`: For the data manipulation
- `Matplotlib`: Doing plotting
- `Numpy`: As a dependency for Pandas
- `Scikit-learn`: The most important library for ML

## ❓ How to run this locally

As the virtualenv for separating the dependencies, I've gone with 
pipenv for it. It's really modular and easy to use.

Use `pipenv shell` to activate the virtualenv and then execute the python
commands to run the files and display accuracy.

## 🤝 Contributing

Contributions, issues and feature requests are welcome. After cloning & setting up project locally, you can just submit a PR to this repo and it will be deployed once it's accepted.

⚠️ It’s good to have descriptive commit messages, or PR titles so that other contributors can understand about your commit or the PR Created.
Read [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.3/) before making the commit message.

And, Also we have a Branch named `dev`, So if you're interested in contributing, Please contribute to that branch instead of the `main` branch.

## 😁 Maintainers

We have 2 maintainers for this project as of now:
- [Sunrit Jana](https://github.com/janaSunrise)
- [Rohith MVK](https://github.com/Rohith04MVK)

## 🙌 Show your support

Be sure to leave a ⭐️ if you like the project!

Made by Sunrit Jana with ❤️
