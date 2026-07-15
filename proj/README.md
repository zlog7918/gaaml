to use library:
- `pip pytorch -e ./proj` (in project's main dir)
- In code:
  ```python
  import gaaml as g
  import numpy as np
  # training_data, test_data: tuple[np.array, np.array]
  # input_num: int
  path='/path/to/empty/dir/for/data'
  ret=g.cr_network(
    training_data,
    test_data,
    save_dir_path=path,
    number_of_attributes=input_num,
  )
  (max_sol,_), _, _=ret.get_statistics()
  model, batch_size, epoch=g.cr_net_from_ind(
    max_sol,
    input_num,
    training_data.shape[0]-input_num
  )
  ```
  gets the most optimal model from run with its parameters of learning

Easiest running with jupyther:
- pip with `requirements.txt` (in project's main dir):
  ```requirements
  pandas
  pytorch
  -e ./proj
  ipykernel
  ipywidgets
  scikit-learn
  ```

Possible to use `tensorflow` instead of `pytorch`:
- `pip tensorflow -e ./proj` (in project's main dir)
- Code before importing gaaml:
  ```python
  import os
  os.environ['KERAS_BACKEND']='tensorflow'
  import gaaml as g
  ```
