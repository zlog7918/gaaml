import typing as t

def _f2f(f: float) -> float: return f

_T=t.TypeVar('_T')
_CTV=t.TypeVar('_CTV', bound=t.Callable)
class MaxAvgMinHolder(t.Generic[_T]):
  __list: list[_T]
  __list_v: list[float]
  @property
  def arr(self) -> list[_T]:
    return self.__list[:self.__curr_len]
  @property
  def arr_v(self) -> list[float]:
    return self.__list_v[:self.__curr_len]
  __v2f: t.Callable[[_T], float]
  @property
  def v2f(self) -> t.Callable[[_T], float]:
    return self.__v2f
  __curr_len: int=0
  __max_i: int=-1
  __min_i: int=-1
  __max_v: float=float('-inf')
  __min_v: float=float('inf')
  __zero_counter: int=0
  __sum: float=0
  def __init__(
    self,
    num: int=10,
    to_val: t.Callable[[_T], float]=_f2f
  ) -> None:
    super().__init__()
    self.__v2f=to_val
    self.__list=[0]*num # type: ignore
    self.__list_v=[0]*num

  def __is_filled_and_expand(self) -> None:
    if self.__curr_len>=len(self.__list):
      self.__list.extend([.0]*len(self.__list)) # type: ignore
      self.__list_v.extend([.0]*len(self.__list))

  def append(self, item: _T) -> None:
    self.__is_filled_and_expand()
    value=self.__v2f(item)
    self.__list[self.__curr_len]=item
    self.__list_v[self.__curr_len]=value
    if value==0:
      self.__zero_counter+=1
    if value>self.__max_v:
      self.__max_i=self.__curr_len
      self.__max_v=value
    if value<self.__min_v:
      self.__min_i=self.__curr_len
      self.__min_v=value
    self.__sum+=value
    self.__curr_len+=1

  def __len__(self):
    return self.__curr_len

  def empty(self) -> bool:
    return self.__curr_len==0

  def __err_if_empty(self) -> None:
    if self.__curr_len==0:
      raise IndexError('No item added yet')

  @property
  def sum(self) -> float:
    return self.__sum
  @property
  def zero_count(self) -> float:
    return self.__zero_counter
  @property
  def max_i(self) -> int:
    self.__err_if_empty()
    return self.__max_i
  @property
  def min_i(self) -> int:
    self.__err_if_empty()
    return self.__min_i
  @property
  def max_v(self) -> float:
    self.__err_if_empty()
    return self.__max_v
  @property
  def avg(self) -> float:
    self.__err_if_empty()
    return self.__sum/self.__curr_len
  @property
  def min_v(self) -> float:
    self.__err_if_empty()
    return self.__min_v
