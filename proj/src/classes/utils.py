from __future__ import annotations
import typing as t
from .Individual import Individual

IndividualType=t.TypeVar('IndividualType', bound=Individual)
CpType=t.TypeVar('CpType')

def _center_of_range(p: int, k: int) -> int:
  return int((k-p)/2)+p

def get_i_in_range(l: list[float], v: float) -> int:
  # In: l=[.3, .4, .6, .8, .9, 1] and v=.87
  # Out: i=4
  # In: l=[.3, .4, .6, .8, .9, 1] and v=.4
  # Out: i=2
  # In: l=[.3, .4, .6, .8, .9, .999] and v=1
  # Out: i=5
  p: int=0
  k: int=len(l)-1
  i: int
  while(p<k):
    i=_center_of_range(p, k)
    if v>l[i]:
      p=i+1
      continue
    if v<l[i]:
      k=i
      continue
    break
  i=_center_of_range(p, k)
  return i

def int_to_bin(value: int, length: int|None) -> str:
  b=bin(value)[2:]
  return b if length is None else b.zfill(length)
