mamba env update -f environment.yml
conda activate toy
pip install pre-commit flake8 black pylint isort
pre-commit install
