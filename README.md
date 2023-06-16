# OptML-2023

Implementation of the OptML Project in Spring 2023 @ EPFL

# Environment Setup

Either using conda, mamba, or micromamba, you can create a virtual environment using the given `environment.yml` file. For example, you could run

```
conda env create -f environment.yml
```

which will create an environment called `OptML2023`. You can then activate the environment by

```
conda activate OptML2023
```

which will allow you to run the code in this repository.

# Repository structure

The repository is structured as follows:

```
.
├── README.md
├── compare_{view}.ipynb
├── data
├── environment.yml
├── output
│   ├── convert.py
│   ├── converted
│   └── {algorithm}_runs_bs{bs}_lr{lr}
│       └ {run_id}
├── plot_utils.py
├── pyproject.toml
├── run
│   ├── README.md
│   ├── config.py
│   ├── datasets.py
│   ├── indexed_dataset.py
│   ├── nns.py
│   ├── run_{algorithm}.py
│   ├── {algorithm}.py
│   └── train_utils.py
└── run.ipynb
```

There are three main notebooks:

- `run.ipynb`: This notebook is used to run the algorithms and save the results in the `output` folder.
- `compare_{view}.ipynb`: This notebook is used to compare the results of the algorithms. The `{view}` parameter can be either `batches` or `snapshots`.

The `data` repository contains the dataset that was used for training and evaluation. If empty, the scripts will download the data as well.

The `output` folder contains

- `convert.py`: a script that converts the output of the algorithms to a format that can be used by the `compare_{view}.ipynb` notebooks (some results were dumped as GPU-loaded tensors).
- `{algorithm}_runs_bs{bs}_lr{lr}`: a folder that contains the results of the algorithm `{algorithm}` with batch size `{bs}` and learning rate `{lr}`. Algorithm could be `saga`, `sgd`, or `svrg` (and could have different parameters), `bs` is 1, 16, 64, 128, and `lr` is fixed to be `1e-2`. Each folder contains 5 folders, numbered from 0 to 4 corresponding to the `run_id`.
- `converted`: a folder that contains the converted results. The naming scheme of the folders in this folder is the same as the top directory.

The `plot_utils.py` file contains some utility methods to plot the metrics. `pyproject.toml` contains some settings for `mypy`.

The `run` folder contains:

- `README.md`: a README file that contains the instructions to run training scripts.
- `config.py`: a shared configuration script for all the algorithms.
- `datasets.py`: a script that contains the dataset download methods.
- `indexed_dataset.py`: a script that contains the `IndexedDataset` class.
- `nns.py`: a script that contains the neural network models.
- `run_{algorithm}.py`: a script that contains the training and evaluation loop for the algorithm `{algorithm}`.
- `{algorithm}.py`: a script that contains the implementation of the algorithm `{algorithm}`.
- `train_utils.py`: a script that contains some utility methods for training and evaluation.
