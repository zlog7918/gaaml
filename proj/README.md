# GAAML (Genetic Algorithm-based AutoML)
GAAML is an AutoML library that uses
a **Genetic Algorithm (GA)** to automatically search
for effective neural network architectures
and training hyperparameters.
It evolves candidate models over multiple generations,
evaluates their performance,
and returns the history of solutions
discovered during the search.

Individuals are evaluated using the output prediction error
between expected outputs and model predictions
on the validation (or training) dataset.
The error is converted into a fitness value,
and the genetic algorithm searches for neural network
architectures and hyperparameters that maximize fitness.

The library currently supports both
**PyTorch** and **TensorFlow** backends.

## Features
* Genetic Algorithm-based neural architecture search
* Automatic hyperparameter optimization
* PyTorch backend (default)
* TensorFlow backend (via Keras backend selection)
* Easy integration into Python projects and Jupyter notebooks

## Workflow

```text
Dataset
   │
   ▼
Genetic Algorithm
   │
   ▼
Best individual
   │
   ▼
Reconstruct model
   │
   ▼
Use trained network
```

## Requirements
- Python **3.10+**
- Tested with:
  - Python 3.10.12
  - Python 3.13.9

## Installation
Clone the repository and install the package from the project root with selected backend.
```bash
git clone <repository-url>
cd <repository-name>
```

### PyTorch (default)
```bash
pip install torch
pip install -e ./proj
```

### TensorFlow backend
```bash
pip install tensorflow
pip install -e ./proj
```

Before importing `gaaml`, select the backend:

```python
import os
os.environ["KERAS_BACKEND"]="tensorflow"

import gaaml as g
```


## Basic Usage
```python
import gaaml as g
import numpy as np

# training_data, test_data: np.ndarray
# input_num: int (>0)

path='/path/to/empty/dir/for/data'

ret=g.cr_network(
  training_data,
  test_data,
  save_dir_path=path,
  number_of_attributes=input_num,
)

(max_sol, _), _, _=ret.get_statistics()

model, batch_size, epoch=g.cr_net_from_ind(
  max_sol,
  input_num,
  training_data.shape[1]-input_num,
)
```

The code above:

1. Runs the genetic optimization process.
2. Collects statistics from the run.
3. Extracts the best evolved individual.
4. Reconstructs the corresponding neural network.
5. Returns:
  - `model` – the best discovered neural network,
  - `batch_size` – the batch size associated with that individual,
  - `epoch` – the recommended number of training epochs.


## Running in Jupyter
For the easiest experience, install the project together with the notebook dependencies.

From the project root:

Save the text below as `requirements.txt`:
```text
pandas
torch
-e ./proj
ipykernel
ipywidgets
scikit-learn
```

Then install:
```bash
pip install -r requirements.txt
```

*The provided `requirements.txt` uses the PyTorch backend.
Replace `torch` with `tensorflow` when using the TensorFlow backend.*

## Data Format
GAAML expects three datasets:
- training data
- validation data (optionally)
- test data

Each dataset is represented as a single table (NumPy array), where the input attributes are followed by the output attributes.

GAAML supports:
- regression tasks, where output columns contain continuous values,
- classification tasks, where output columns should be one-hot encoded.

For example, for a problem with six input features and four outputs:

|    x₁ |    x₂ |    x₃ |    x₄ |    x₅ |    x₆ |    y₁ |    y₂ |    y₃ |    y₄ |
| ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 0.767 | 0.352 | 0.663 | 0.876 | 0.612 | 0.094 | 0.287 | 0.218 | 0.626 | 0.369 |
| 0.670 | 0.886 | 0.535 | 0.031 | 0.073 | 0.239 | 0.519 | 0.723 | 0.639 | 0.717 |
|     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |

For classification tasks, output columns should use one-hot encoding:

|    x₁ |    x₂ |    x₃ |    x₄ |    x₅ |    x₆ | y₁ | y₂ | y₃ | y₄ |
| ----: | ----: | ----: | ----: | ----: | ----: | -: | -: | -: | -: |
| 0.767 | 0.352 | 0.663 | 0.876 | 0.612 | 0.094 |  0 |  0 |  1 |  0 |
| 0.670 | 0.886 | 0.535 | 0.031 | 0.073 | 0.239 |  1 |  0 |  0 |  0 |
|     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |     ⋮ |  ⋮ |  ⋮ |  ⋮ |  ⋮ |

`number_of_attributes` specifies the number of input features (`x` columns). The remaining columns are treated as output variables.
If `number_of_attributes` is negative its absolute value specifies number of output variables (`y` columns).

### Validation Data

Validation data is optional.

When provided, validation data is used during the genetic algorithm search to evaluate individuals and calculate their fitness. This allows the algorithm to select neural network architectures and hyperparameters that generalize better to unseen data.

With validation data:
```python
ret=g.cr_network(
  training_data,
  validation_data,
  test_data,
  save_dir_path=path,
  number_of_attributes=input_num,
)
```

Without validation data:
```python
ret=g.cr_network(
  training_data,
  test_data,
  save_dir_path=path,
  number_of_attributes=input_num,
)
```

If validation data is not provided,
the training dataset is used for fitness evaluation.
The test dataset is only used for final
evaluation of the discovered models.

## Output
`cr_network()` returns an optimization result object containing the evolutionary history and statistics of the search.

The best individual can be obtained using:

```python
(max_sol, _), _, _ = ret.get_statistics()
```

and reconstructed into a trainable neural network:

```python
model, batch_size, epoch = g.cr_net_from_ind(
    max_sol,
    input_num,
    output_num,
)
```

where:
- `input_num` is the number of input attributes,
- `output_num` is the number of output attributes,
- `model` is the best evolved neural network,
- `batch_size` is its batch size,
- `epoch` is the corresponding number of training epochs.


## Saved Files
`save_dir_path` specifies the directory used to store the results of the evolutionary search. It will contain the trained model weights produced during the optimization process, allowing the discovered models to be reconstructed without retraining.

It may either be empty or not exist before running GAAML.


## Notes
- The default backend is **PyTorch**.
- To use TensorFlow, set the `KERAS_BACKEND` environment variable **before** importing `gaaml`.
- The project should support other backends (except NumPy) available in keras, although this has not been tested.
- The project **does not** support NumPy backend as it does not support model fitting.

## License
GNU General Public License v3.0
