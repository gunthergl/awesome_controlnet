<!-- [![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/) -->
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/cc_pipeline.svg?branch=main)](https://cirrus-ci.com/github/<USER>/cc_pipeline)
[![ReadTheDocs](https://readthedocs.org/projects/cc_pipeline/badge/?version=latest)](https://cc_pipeline.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/cc_pipeline/main.svg)](https://coveralls.io/r/<USER>/cc_pipeline)
[![PyPI-Server](https://img.shields.io/pypi/v/cc_pipeline.svg)](https://pypi.org/project/cc_pipeline/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/cc_pipeline.svg)](https://anaconda.org/conda-forge/cc_pipeline)
[![Monthly Downloads](https://pepy.tech/badge/cc_pipeline/month)](https://pepy.tech/project/cc_pipeline)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/cc_pipeline)
-->

# cc_pipeline

> This pipeline packages ControlNet in a fastAPI usecase.

# HowTo develop

1. Clone the repository
2. Create a new branch
3. Develop
   1. Install pre-commit (and use it)
   2. Install the environment
   3. Use "tox" to install the package and run the tests
4. Create a pull-request to the main branch, you cannot push directly to the main branch.

## Installation

In order to set up the necessary environment:

```
conda env create -f environment.yml
conda activate cc_pipeline
```


Then take a look into the `scripts` and `notebooks` folders.

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── src
│   └── cc_pipeline         <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject


## How was the package set up?

```bash
# Install PyScaffold
pip install pyscaffold
pip install pyscaffoldext-dsproject

# Create a new project
putup --dsproject cc_pipeline --description "This pipeline packages ControlNet in a fastAPI usecase."

```

Copy the relevant packages into environment.yml.

```bash

conda env create -f environment.yml
conda activate cc_pipeline

# Set up precommit
pre-commit install
### Autoupdate introduced problems with python version, isort and black, so I leave that here for now.
# # [WARNING] The 'rev' field of repo 'https://github.com/psf/black' appears to be a mutable reference (moving tag / branch).  Mutable references are never updated after first install and are not supported.  See https://pre-commit.com/#using-the-latest-version-for-a-repository for more details.  Hint: `pre-commit autoupdate` often fixes this.
# pre-commit autoupdate

### Check .pre-commit-config.yaml, then commit

### Remove irrelevant folders and files
```


Get ControlNet


```bash
cd src/cc_pipeline
git clone https://github.com/lllyasviel/ControlNet.git
# remove irrelevant files
rm -rf ControlNet
mv ControlNet/* .
rm tutorial*
rm gradio*
rm -r annotator font github_page docs test_imgs
rm tool_*
```
