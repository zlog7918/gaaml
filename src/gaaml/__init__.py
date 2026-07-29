import os
if os.environ.get('KERAS_BACKEND') is None:
  os.environ['KERAS_BACKEND']='torch'
from .core import cr_network
from . import consts as const
from .utils import cr_net_from_ind
from .classes.Generations import Generations
from .classes.NetIndividual import NetIndividual

__all__=[
  'const',
  'cr_network',
  'Generations',
  'NetIndividual',
  'cr_net_from_ind',
]
