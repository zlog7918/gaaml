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
}

__keras_optimalization_types: dict[int, type[krs.optimizers.Optimizer]]={
  0: krs.optimizers.SGD,
  1: krs.optimizers.Adam,
}

# __keras_losses_types: dict[int, type[krs.losses.Loss]]={
#   0: krs.losses.MeanSquaredError,
#   # 1: krs.losses.,
# }

def __cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int) -> krs.models.Model:
  # TODO
  params, layer_sizes, layer_types=net_ind.gen
  params, layer_sizes, layer_types=params.fenotype, layer_sizes.fenotype, layer_types.fenotype

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
def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int) -> tuple[krs.models.Model, int, int]:
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
  model=__cr_net_from_ind(net_ind, input_size, output_size)
  return (model, params[const.BIN_PART_BATCH_NAME], params[const.BIN_PART_EPOCHS_NAME])

def __is_categorical(arr: np.ndarray) -> bool|None:
  pass

count: int=0
def get_fit_func(
  training_data: np.ndarray,
  validation_data: np.ndarray|None,
  test_data: np.ndarray,
  number_of_attributes: int,
  # categorial: bool|None=None,
) -> t.Callable[[NetIndividual, Path], float]:
  if training_data.shape[1]!=test_data.shape[1]:
    raise ValueError('training_data and test_data do not have the same number of attributes in data or output')
  if validation_data is not None and training_data.shape[1]!=validation_data.shape[1]:
    raise ValueError('validation_data does not have the same number of attributes in data or output as training_data and test_data')
  training_data_x, training_data_y=training_data[:,:number_of_attributes], training_data[:,number_of_attributes:]
  assumed_categorial=None
  # if training_data_x.shape[1]==1 and :
  #   categorial=False
  # else:
  #   categorial=False
  # if categorial is None:
  #   categorial
  _validation_data=None if validation_data is None else (validation_data[:,:number_of_attributes], validation_data[:,number_of_attributes:])
  del validation_data
  test_data_x, test_data_y=test_data[:,:number_of_attributes], test_data[:,number_of_attributes:]

  def f(
    net_ind: NetIndividual,
    dir: Path,
    verbose: t.Literal[0]|t.Literal[1]|t.Literal[2]|t.Literal['auto']='auto',
  ) -> float:
    global count
    count+=1
    model, batch_size, epochs=cr_net_from_ind(net_ind, training_data_x.shape[1], training_data_y.shape[1])
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
    with warnings.catch_warnings(category=DeprecationWarning):
      warnings.filterwarnings(
        'ignore',
        message='__array__ implementation doesn\'t accept a copy keyword',
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