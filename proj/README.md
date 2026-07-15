# GAAML (Genetic Algorithm-based AutoML)
GAAML is an AutoML library that uses
a **Genetic Algorithm (GA)** to automatically search
for effective neural network architectures
and training hyperparameters.
It evolves candidate models over multiple generations,
evaluates their performance,
and returns the history of solutions
discovered during the search.

The library currently supports both
**PyTorch** and **TensorFlow** backends.

## Features

* Genetic Algorithm-based neural architecture search
* Automatic hyperparameter optimization
* PyTorch backend (default)
* TensorFlow backend (via Keras backend selection)
* Easy integration into Python projects and Jupyter notebooks

---

## Installation

### PyTorch (default)
Install PyTorch together with the library.
From the project's root directory:
```bash
pip install torch -e ./proj
```

---

### TensorFlow backend
Install TensorFlow together with the library.
From the project's root directory:
```bash
pip install tensorflow -e ./proj
```

Before importing `gaaml`, select the TensorFlow backend:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import gaaml as g
```

---

## Basic Usage

```python
import gaaml as g
import numpy as np

# training_data, test_data: tuple[np.ndarray, np.ndarray]
# input_num: int

path = "/path/to/empty/dir/for/data"

ret = g.cr_network(
    training_data,
    test_data,
    save_dir_path=path,
    number_of_attributes=input_num,
)

(max_sol, _), _, _ = ret.get_statistics()

model, batch_size, epoch = g.cr_net_from_ind(
    max_sol,
    input_num,
    training_data.shape[0] - input_num,
)
```

The code above:

1. Runs the genetic optimization process.
2. Collects statistics from the run.
3. Extracts the best evolved individual.
4. Reconstructs the corresponding neural network.
5. Returns:

   * `model` – the best discovered neural network,
   * `batch_size` – the batch size associated with that individual,
   * `epoch` – the recommended number of training epochs.

---

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

---

## Notes

* `save_dir_path` should point to an **empty or not-existent directory** where GAAML can store intermediate results and generated data.
* The default backend is **PyTorch**.
* To use TensorFlow, set the `KERAS_BACKEND` environment variable **before** importing `gaaml`.
* Project should support other backends (except NumPy) available in keras, however it has not been tested.
* Project **does not** support NumPy backend as it does not support model fitting.

<!-- --- -->
<!-- ## License -->
<!-- Add your project's license here. -->
