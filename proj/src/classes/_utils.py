import typing as t
import random as rnd
from .Individual import Individual

ord0=ord('0')
ord1=ord('1')
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

def correct_gen_to_min_max(gen: bytearray, min_v: int, max_v: int) -> int:
  l=len(gen)
  gen=gen.copy()
  max_gen=int_to_bin(max_v-min_v, l)
  correct_gens: t.Callable[[bytearray, bytearray], bool]=lambda g, max_g: int(g, base=2)<=int(max_g, base=2)
  if not correct_gens(gen, max_gen):
    for i, org_1gen, max_1gen in zip(range(l), gen, max_gen):
      if org_1gen==max_1gen:
        continue
      # gen[i]=0
      gen[i]=ord0
      # gen=f'{gen[:i]}0{gen[i+1:]}'
      if correct_gens(gen, max_gen):
        break
  return int(gen, base=2)+min_v

def randint(a: int, b: int) -> int:
  return rnd.randint(a, b) if a!=b else a

def int_to_bin(value: int, length: int|None) -> bytearray:
  b=bin(value)[2:]
  b=bytearray(
    # int(el)
    ord(el)
      for el in
    (
      b
        if length is None else
      b.zfill(length)
    )
  )
  return b
