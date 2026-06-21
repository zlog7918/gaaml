import os
if os.environ.get('KERAS_BACKEND') is None:
  os.environ['KERAS_BACKEND']='torch'
from .core import cr_network

__all__=["cr_network"]
