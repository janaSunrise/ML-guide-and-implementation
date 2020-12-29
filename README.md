# Sklearn dataset predictions

This repository contains the predictions, and plots 
for the datasets included in the scikit learn library 
by default and also some other datasets from kaggle or other sources.

## 🛠️ Tech stack used

- `Pandas`: For the data manipulation
- `Matplotlib`: Doing plotting
- `Numpy`: As a dependency for Pandas
- `Scikit-learn`: The most important library for ML

## ❓ How to run this locally

### NOTE:

Before cloning this repo, you need to ensure you have [GIT LFS](https://git-lfs.github.com/) 
installed on your local system. Because this repository contains several `*.csv` files, 
which are quite large and aren't accepted by github directly. Sorry for this inconvience.

### Steps for running locally:

- Run for Testing

  As the virtualenv for separating the dependencies, I've gone with 
  pipenv for it. It's really modular and easy to use.
  
  Use `pipenv shell` to activate the virtualenv and then execute the python
  commands to run the files and display accuracy.

- Run for development and contributing

  We also encourage people to support this repository by contributing, and keeping it alive.
  But note that we follow certain steps to ensure code is clean, organized and readable using
  linting with `flake8`. We also encourage using pre-commit for pushing clean code.

  Steps to set up:
  - Install dependencies: `pipenv update -d`
  - Setup pre commit: `pipenv run precommit`
  - After changes, try linting: `pipenv run lint`

## Datasets implemented

### Diabetes: 

This dataset consists of 9 columns.
The target value which has to be predicted is `diabetes`
This is a classifier problem, where the value of diabetes in boolean,
but in integer format.

Algorithm used for the problem: `GradientBoostingClassifier`

Accuracy achieved: `0.74`

## 🤝 Contributing

Contributions, issues and feature requests are welcome. After cloning 
& setting up project locally, you can just submit a PR to this 
repo and it will be deployed once it's accepted. The contributing 
file can be found [here](https://github.com/janaSunrise/sklearn-datasets-implementation/blob/main/CONTRIBUTING.md).

⚠️ It’s good to have descriptive commit messages, or PR titles so that other contributors can understand about your commit or the PR Created.
Read [conventional commits](https://www.conventionalcommits.org/en/v1.0.0-beta.3/) before making the commit message.

And, for contributions we have a Branch named `dev`, So if you're interested in contributing, 
Please contribute to that branch instead of the `main` branch.

## 😁 Maintainers

We have 2 maintainers for this project as of now:
- [Sunrit Jana](https://github.com/janaSunrise)
- [Rohith MVK](https://github.com/Rohith04MVK)

## 🙌 Show your support

Be sure to leave a ⭐️ if you like the project, and also be sure to contribute, if you're interested!

<div align="center">

Made by Sunrit Jana with ❤️

</div>
