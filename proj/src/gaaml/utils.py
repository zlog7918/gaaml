import json
import warnings
import numpy as np
import typing as t
import keras as krs
from pathlib import Path
from . import consts as const
from .classes.NetIndividual import NetIndividual

__keras_activation_types: dict[int, t.Callable]={
  0: krs.activations.relu,
  1: krs.activations.tanh,
  2: krs.activations.sigmoid,
  3: krs.activations.leaky_relu,
  -1: krs.activations.softmax,
}

__keras_optimalization_types: dict[int, type[krs.optimizers.Optimizer]]={
  0: krs.optimizers.SGD,
  1: krs.optimizers.Adam,
}

# __keras_losses_types: dict[int, type[krs.losses.Loss]]={
#   0: krs.losses.MeanSquaredError,
#   # 1: krs.losses.,
# }

def __cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int, categorial: bool) -> krs.models.Model:
  # TODO
  params, layer_sizes, layer_types=net_ind.gen
  params, layer_sizes, layer_types=params.fenotype, layer_sizes.fenotype, layer_types.fenotype
  if categorial:
    layer_types[-1]=-1

  seq=krs.models.Sequential()
  seq.add(krs.layers.Input(shape=(input_size,)))
  for n, t in zip((*layer_sizes, output_size), layer_types):
    seq.add(krs.layers.Dense(n, activation=__keras_activation_types[t]))
  learning_rate=params[const.BIN_PART_LEARNING_RATE_NAME]/(2<<13)
  seq.compile(
    __keras_optimalization_types[params[const.BIN_PART_OPTIMIZER_NAME]](
      learning_rate=learning_rate
    ),
    krs.losses.MeanSquaredError(),
  )
  return seq

# def swith_backend(backend: t.Literal['torch']|t.Literal['tensorflow']) -> None: ...
#   # TODO:
#   # Is possible to switch between tensorflow and torch using:
#   # krs.config.set_backend()
def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int, categorial: bool) -> tuple[krs.models.Model, int, int]:
# def cr_net_from_ind(
#   net_ind: NetIndividual,
#   input_size: int,
#   output_size: int,
#   backend: t.Literal['torch']|t.Literal['tensorflow']='tensorflow',
# ) -> tuple[krs.models.Model, int, int]:
#   if krs.config.backend()!=backend:
#     swith_backend(backend)
  params, _, _=net_ind.gen
  params=params.fenotype
  model=__cr_net_from_ind(net_ind, input_size, output_size, categorial)
  return (model, params[const.BIN_PART_BATCH_NAME], params[const.BIN_PART_EPOCHS_NAME])

def __detect_type_group(arr: np.ndarray) -> str|None:
  flat=arr.flatten()

  is_str=True
  is_bool=True
  is_int=True
  is_int01=True

  for x in flat:
    if is_str and not isinstance(x, (str, np.str_)):
      is_str=False
    if is_bool and not isinstance(x, (bool, np.bool_)):
      is_bool=False
    if is_int:
      if not (
        isinstance(x, (int, np.integer))
        and not isinstance(x, (bool, np.bool_))
      ):
        is_int=False
        is_int01=False
      elif is_int01 and x not in {0, 1}:
        is_int01=False
    if not (is_str or is_bool or is_int):
      return None

  if is_str:
    return 'str'
  if is_bool:
    return 'bool'
  if is_int01:
    return 'int01'
  if is_int:
    return 'int'
  return None

def __is_categorial(arr: np.ndarray) -> bool|None:
  detected_type=__detect_type_group(arr)
  if detected_type is None:
    return False
  # if arr.shape[1]==1:
  #   if detected_type=='str': # all strings -> categorial
  #     return True
  #   if detected_type in {'int', 'int01'}: # all integers -> may be categorial
  #     return None
  #   return False
  if arr.shape[1]!=1 and detected_type in {'bool', 'int01'}:
    row_sums=np.sum(arr, axis=1)
    if np.all(row_sums==1):
      return True if detected_type=='bool' else None
  # anything else -> not categorial
  return False

_P=t.TypeVar('_P')
_T=t.TypeVar('_T')
def __into_tuple(*args: tuple[_P|None, t.Callable[[_P], _T]]|tuple[_T|None]) -> tuple[_T, ...]:
  return tuple(a[0] if len(a)==1 else a[1](a[0]) for a in args if a[0] is not None)

count: int=0
def get_fit_func(
  training_data: np.ndarray,
  validation_data: np.ndarray|None,
  test_data: np.ndarray,
  number_of_attributes: int,
  categorial: bool=False,
) -> t.Callable[[NetIndividual, Path], float]:
  if any(data.ndim!=2 for data in __into_tuple(
    (training_data,),
    (validation_data,),
    (test_data,),
  )):
  # if training_data.ndim!=2 or (validation_data is not None and validation_data.ndim!=2) or test_data.ndim!=2:
    raise ValueError('Every data must be 2D array')
  if training_data.shape[1]!=test_data.shape[1]:
    raise ValueError('training_data and test_data do not have the same number of attributes in data or output')
  if validation_data is not None and training_data.shape[1]!=validation_data.shape[1]:
    raise ValueError('validation_data does not have the same number of attributes in data or output as training_data and test_data')
  # print(training_data)
  _training_data=training_data[:,:number_of_attributes], training_data[:,number_of_attributes:]
  training_data_x, training_data_y=(
    np.asarray(
      _training_data[0],
      dtype=np.dtypes.Float64DType,
    ),
    np.asarray(
      _training_data[1],
      dtype=np.dtypes.Int8DType if categorial else np.dtypes.Float64DType,
    ),
  )
  # assert _training_data[0].all(training_data_x)
  assert (_training_data[0]==training_data_x).all()
  assert (_training_data[1]==training_data_y).all()
  del _training_data, training_data
  _validation_data=None
  if validation_data is not None:
    __validation_data=validation_data[:,:number_of_attributes], validation_data[:,number_of_attributes:]
    _validation_data=(
      np.asarray(
        __validation_data[0],
        dtype=np.dtypes.Float64DType,
      ),
      np.asarray(
        __validation_data[1],
        dtype=np.dtypes.Int8DType if categorial else np.dtypes.Float64DType,
      ),
    )
    assert (__validation_data[0]==_validation_data[0]).all()
    assert (__validation_data[1]==_validation_data[1]).all()
    del __validation_data
  del validation_data

  _test_data=test_data[:,:number_of_attributes], test_data[:,number_of_attributes:]
  test_data_x, test_data_y=(
    np.asarray(
      _test_data[0],
      dtype=np.dtypes.Float64DType,
    ),
    np.asarray(
      _test_data[1],
      dtype=np.dtypes.Int8DType if categorial else np.dtypes.Float64DType,
    ),
  )
  assert (_test_data[0]==test_data_x).all()
  assert (_test_data[1]==test_data_y).all()
  del _test_data, test_data, number_of_attributes
  assumed_categorial=__is_categorial(np.concatenate(
    __into_tuple(
      (training_data_y,),
      (_validation_data, lambda x: x[1]),
      (test_data_y,),
    ),
    axis=0,
  ))
  if assumed_categorial is not None and categorial!=assumed_categorial:
    raise ValueError('Given type is not valid for given data')
  del assumed_categorial
  def f(
    net_ind: NetIndividual,
    dir: Path,
    verbose: t.Literal[0]|t.Literal[1]|t.Literal[2]|t.Literal['auto']='auto',
  ) -> float:
    global count
    count+=1
    model, batch_size, epochs=cr_net_from_ind(
      net_ind,
      training_data_x.shape[1],
      training_data_y.shape[1],
      categorial
    )
    with warnings.catch_warnings():
      warnings.filterwarnings(
        'ignore',
        # message='__array__ implementation doesn\'t accept a copy keyword',
        category=DeprecationWarning,
      )
      fit_ret=model.fit(
        training_data_x,
        training_data_y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=_validation_data,
        verbose=verbose, # type: ignore
      ) # throws 3 warnings: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed.

    with open(dir/'model_meta.data', 'x') as meta:
      json.dump(
        {
          'id': count,
          'epoch': epochs,
          'batch': batch_size,
          'backend': krs.config.backend(),
          'hidden_len': len(model.get_weights())//2-1,
          'fit_ret': str(fit_ret),
        },
        meta,
      )
    with warnings.catch_warnings():
      warnings.filterwarnings(
        'ignore',
        # message='__array__ implementation doesn\'t accept a copy keyword',
        category=DeprecationWarning,
      )
      model.save_weights(dir/'model.weights.h5')

      ret=model.evaluate(
        test_data_x,
        test_data_y,
        verbose=verbose, # type: ignore
      )
    return ret
  return f