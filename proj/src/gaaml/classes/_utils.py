import typing as t
import random as rnd
from .Individual import _BaseIndividual

ord0=ord('0')
ord1=ord('1')
IndividualType=t.TypeVar('IndividualType', bound=_BaseIndividual)
BitSize_Min_Max: t.TypeAlias=tuple[int, int, int]

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
  n: int=len(l)
  k: int=n-1
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
  if l[i]==v and i!=n-1:
    i+=1
  return i

def __get_i_maxs(a: bytearray, b: bytearray) -> set[int]:
  return {i for i, (ai, bi) in enumerate(zip(a,b)) if ai>bi}
def correct_gen_to_min_max(gen: bytearray, min_v: int, max_v: int) -> int:
  l=len(gen)
  m=int_to_bin(max_v-min_v, l)
  m_g=min(__get_i_maxs(m, gen), default=l)
  g_m=__get_i_maxs(gen[:m_g], m[:m_g])
  gen=bytearray(ord0 if i in g_m else gen[i] for i in range(l))
  return int(gen, base=2)+min_v

def randint(a: int, b: int) -> int:
  return rnd.randint(a, b) if a!=b else a

def int_to_bin(value: int, length: int|None) -> bytearray:
  b=bin(value)[2:]
  b=bytearray(
    ord(el)
      for el in
    (
      b
        if length is None else
      reversed(tuple(reversed(b.zfill(length)))[:length])
    )
  )
  return b
