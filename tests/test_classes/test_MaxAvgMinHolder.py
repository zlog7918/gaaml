import pytest
import typing as t
from gaaml.classes.MaxAvgMinHolder import (
  _f2f as f2f,
  MaxAvgMinHolder as MAMH,
)

mark__test___f2f=pytest.mark.parametrize(
  'v',
  [3, 5., 2., 65., 2., 6],
)
@mark__test___f2f
def test___f2f(v: float) -> None:
  # values ^

  # test
  ret=f2f(v)

  # results
  assert ret==v

def test_empty_initial_state() -> None:
  # values
  holder=MAMH()

  # test
  length=len(holder)
  is_empty=holder.empty()
  total=holder.sum
  zeros=holder.zero_count
  arr=holder.arr
  arr_v=holder.arr_v
  v2f=holder.v2f

  # results
  assert length==0
  assert is_empty is True
  assert total==0
  assert zeros==0
  assert arr==[]
  assert arr_v==pytest.approx([])
  assert isinstance(v2f, staticmethod)
  assert isinstance(v2f.__wrapped__, t.Callable)
  assert v2f.__wrapped__ is f2f

def test_empty_initial_state_own_func() -> None:
  # values
  v2f=lambda v: sum(v)/2
  holder=MAMH[tuple[float, float]](to_val=v2f)

  # test
  length=len(holder)
  is_empty=holder.empty()
  total=holder.sum
  zeros=holder.zero_count
  arr=holder.arr
  arr_v=holder.arr_v
  ret_v2f=holder.v2f

  # results
  assert length==0
  assert is_empty is True
  assert total==0
  assert zeros==0
  assert arr==[]
  assert arr_v==pytest.approx([])
  assert isinstance(ret_v2f, staticmethod)
  assert isinstance(ret_v2f.__wrapped__, t.Callable)
  assert ret_v2f.__wrapped__ is not f2f
  assert ret_v2f.__wrapped__ is v2f

def test_single_append() -> None:
  # values
  holder=MAMH()
  value=5.

  # test
  holder.append(value)

  # results
  assert len(holder)==1
  assert holder.empty() is False
  assert holder.sum==pytest.approx(5.)
  assert holder.avg==pytest.approx(5.)
  assert holder.max_v==pytest.approx(5.)
  assert holder.min_v==pytest.approx(5.)
  assert holder.max_i==0
  assert holder.min_i==0
  assert holder.zero_count==0
  assert holder.arr==[5.]
  assert holder.arr_v==pytest.approx([5.])
  assert holder.arr_v==pytest.approx([*map(holder.v2f, holder.arr)])

def test_single_append_own_func() -> None:
  # values
  v2f=lambda v: sum(v)/3
  holder=MAMH[tuple[float, int, float]](to_val=v2f)
  item=(5., 6, 1)
  exp_val=v2f(item)

  # test
  holder.append(item)

  # results
  assert len(holder)==1
  assert holder.empty() is False
  assert holder.sum==pytest.approx(exp_val)
  assert holder.avg==pytest.approx(exp_val)
  assert holder.max_v==pytest.approx(exp_val)
  assert holder.min_v==pytest.approx(exp_val)
  assert holder.max_i==0
  assert holder.min_i==0
  assert holder.zero_count==0
  assert holder.arr==[item]
  assert holder.arr_v==pytest.approx([exp_val])
  assert holder.arr_v==[*map(holder.v2f, holder.arr)]

def test_multiple_appends() -> None:
  # values
  holder=MAMH()
  values=[1., 3., -2., 7., 4.]

  # test
  for v in values:
    holder.append(v)

  # results
  assert len(holder)==len(values)
  assert holder.sum==pytest.approx(sum(values))
  assert holder.avg==pytest.approx(sum(values)/len(values))
  assert holder.max_v==7.
  assert holder.min_v==-2.
  assert holder.max_i==values.index(7.)
  assert holder.min_i==values.index(-2.)
  assert holder.zero_count==0
  assert holder.arr==values
  assert holder.arr_v==values
  assert holder.arr_v==[*map(holder.v2f, holder.arr)]

def test_multiple_appends_own_func() -> None:
  # values
  v2f=lambda v: sum(v)/3
  holder=MAMH[tuple[float, int, float]](to_val=v2f)
  items=[
    (1., 0, 2.),
    (2., 6, 1.),
    (-2.5, -4, .5),
    (7., 6, 8),
    (4., 4, 4),
  ]

  # test
  exp_val=[]
  for i in items:
    holder.append(i)
    exp_val.append(v2f(i))

  # results
  assert len(holder)==len(items)
  assert len(holder)==len(exp_val)
  assert holder.sum==pytest.approx(sum(exp_val))
  assert holder.avg==pytest.approx(sum(exp_val)/len(exp_val))
  assert holder.max_v==pytest.approx(7.)
  assert holder.min_v==pytest.approx(-2.)
  assert holder.max_i==items.index((7., 6, 8))
  assert holder.min_i==items.index((-2.5, -4, .5))
  assert holder.zero_count==0
  assert holder.arr==items
  assert holder.arr_v==[*map(v2f, holder.arr)]

def test_zero_count() -> None:
  # values
  holder=MAMH()
  values=[0., 1., 0., 2., 0.]
  for v in values:
    holder.append(v)

  # test
  zero_count=holder.zero_count

  # results
  assert len(holder)==5
  assert zero_count==3

def test_zero_count_own_func() -> None:
  # values
  v2f=lambda v: v[1]
  holder=MAMH[tuple[float, int]](to_val=v2f)
  items=[
    (1., 0),
    (2., 0),
    (-2.5, -4),
    (7., 0),
    (4., 4),
  ]
  for i in items:
    holder.append(i)

  # test
  zero_count=holder.zero_count

  # results
  assert len(holder)==5
  assert zero_count==3

def test_dynamic_resize() -> None:
  # values
  holder=MAMH(num=2)
  values=[1., 2., 3., 4.]
  for v in values:
    holder.append(v)

  # test
  arr=holder.arr
  max_v=holder.max_v
  min_v=holder.min_v

  # results
  assert len(holder)==len(values)
  assert len(arr)==len(values)
  assert arr==values
  assert max_v==pytest.approx(4.)
  assert min_v==pytest.approx(1.)

def test_all_equal_values() -> None:
  # values
  holder=MAMH()
  values=[5., 5., 5.]
  for v in values:
    holder.append(v)

  # test
  max_v=holder.max_v
  min_v=holder.min_v
  max_i=holder.max_i
  min_i=holder.min_i
  avg=holder.avg

  # results
  assert max_v==pytest.approx(5.)
  assert min_v==pytest.approx(5.)
  assert max_i==0
  assert min_i==0
  assert avg==pytest.approx(5.)

def test_negative_values() -> None:
  # values
  holder=MAMH()
  values=[-10., -5., -20.]
  for v in values:
    holder.append(v)

  # test
  max_v=holder.max_v
  min_v=holder.min_v
  max_i=holder.max_i
  min_i=holder.min_i
  _sum=holder.sum
  avg=holder.avg

  # results
  assert max_v==pytest.approx(-5.)
  assert min_v==pytest.approx(-20.)
  assert max_i==1
  assert min_i==2
  assert _sum==pytest.approx(sum(values))
  assert avg==pytest.approx(sum(values)/len(values))

def test_arr_and_arr_v_return_copys1() -> None:
  # values
  holder=MAMH()
  holder.append(1.)
  holder.append(2.)

  # test
  arr=holder.arr
  arr_v=holder.arr_v
  arr.append(999.)
  arr_v.append(959.)

  # results
  assert holder.arr==[1., 2.]
  assert holder.arr_v==pytest.approx([1., 2.])
  assert arr==[1., 2., 999.]
  assert arr_v==pytest.approx([1., 2., 959.])

def test_arr_and_arr_v_return_copys2() -> None:
  # values
  holder=MAMH()
  holder.append(1.)
  holder.append(2.)

  # test
  arr=holder.arr
  arr_v=holder.arr_v
  holder.append(999.)

  # results
  assert holder.arr==[1., 2., 999.]
  assert holder.arr_v==pytest.approx([1., 2., 999.])
  assert arr==[1., 2.]
  assert arr_v==pytest.approx([1., 2.])

mark__test_empty_raises_on_properties=pytest.mark.parametrize(
  'prop',
  [
    MAMH.max_i,
    MAMH.min_i,
    MAMH.max_v,
    MAMH.min_v,
    MAMH.avg,
  ],
)
@mark__test_empty_raises_on_properties
def test_empty_raises_on_properties(prop: property) -> None:
  # values
  holder=MAMH()
  exp_str='No item added yet'

  # # test
  with pytest.raises(IndexError) as excinfo:
    _=prop.__get__(holder)

  # results
  assert str(excinfo.value)==exp_str
