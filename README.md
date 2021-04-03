# beer-recommender


## Notebooks

To fire up the notebooks we recommend using [JupyterLab](http://jupyterlab.io/) within a virtualenv.

This was tested in Python 3.8.

To create a virtualenv:

- `python -m venv .venv`

To activate it:

- `source .venv/bin/activate`

To install dependencies:

  a. The vanilla way:
  - `pip install -r requirements.txt`

  b. The pip-tools way:
  1. `pip install --upgrade pip`
  2. `pip install pip-tools`
  3. `pip-sync`

  4. To change any requirements:

     - Change them manually in `requirements.in`
     - Do `pip-compile` -> This will update `requirements.txt` automatically.
