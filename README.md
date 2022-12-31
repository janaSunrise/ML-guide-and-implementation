# ML guide & implementation

Perfect guide for learning about machine learning algorithms along w/ implementations
in python-based frameworks.

## How to refer?

The repository is divided into 2 main components:
- Machine learning
- Deep learning

Further, there are nested directories. Each containing a folder for each algorithm (in machine learning)
or Framework (for deep learning) with a `README.md` file containing the guide for the algorithm along with
nested folders demonstrating basic implementations.

## How to run the code?

You'll need to have Jupyter notebooks setup. If you do, you're good, else you can use VSCode directly to
run and view those.

Install the pipenv dependencies using `pipenv install` and then run `pipenv shell` to activate the shell.

First, every folder has a `setup.sh` if it uses an external dataset, to setup the data. Run that with the
`pipenv` environment activated.

Finally, you can run the notebooks using `jupyter notebook`, or in VSC.
