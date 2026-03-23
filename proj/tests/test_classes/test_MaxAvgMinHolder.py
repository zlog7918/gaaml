import pytest
from gaaml.classes.MaxAvgMinHolder import MaxAvgMinHolder as MAMH

def test_empty_initial_state() -> None:
  # values
  holder=MAMH()

  # test
  length=len(holder)
  is_empty=holder.empty()
  total=holder.sum
  zeros=holder.zero_count
  arr=holder.arr

  # results
  assert length==0
  assert is_empty is True
  assert total==0
  assert zeros==0
  assert arr==[]

def test_empty_raises_on_properties() -> None:
  # values
  holder=MAMH()
  exp_str='No item added yet'

  # test
  with pytest.raises(IndexError) as excinfo1:
    _=holder.max_i
  with pytest.raises(IndexError) as excinfo2:
    _=holder.min_i
  with pytest.raises(IndexError) as excinfo3:
    _=holder.max_v
  with pytest.raises(IndexError) as excinfo4:
    _=holder.min_v
  with pytest.raises(IndexError) as excinfo5:
    _=holder.avg

  # results
  assert all(str(excinfo.value)==exp_str for excinfo in (excinfo1, excinfo2, excinfo3, excinfo4, excinfo5))

def test_single_append() -> None:
  # values
  holder=MAMH()
  value=5.0

  # test
  holder.append(value)

  # results
  assert len(holder)==1
  assert holder.empty() is False
  assert holder.sum==5.0
  assert holder.avg==5.0
  assert holder.max_v==5.0
  assert holder.min_v==5.0
  assert holder.max_i==0
  assert holder.min_i==0
  assert holder.zero_count==0
  assert holder.arr==[5.0]


def test_multiple_appends() -> None:
  # values
  holder=MAMH()
  values=[1.0, 3.0, -2.0, 7.0, 4.0]

  # test
  for v in values:
    holder.append(v)

  # results
  assert len(holder)==len(values)
  assert holder.sum==sum(values)
  assert holder.avg==sum(values)/len(values)
  assert holder.max_v==7.0
  assert holder.min_v==-2.0
  assert holder.max_i==values.index(7.0)
  assert holder.min_i==values.index(-2.0)
  assert holder.arr==values


def test_zero_count() -> None:
  # values
  holder=MAMH()
  values=[0.0, 1.0, 0.0, 2.0, 0.0]

  # test
  for v in values:
    holder.append(v)

  # results
  assert holder.zero_count==3


def test_dynamic_resize() -> None:
  # values
  holder=MAMH(num=2)
  values=[1.0, 2.0, 3.0, 4.0]

  # test
  for v in values:
    holder.append(v)

  # results
  assert len(holder)==4
  assert holder.arr==values
  assert holder.max_v==4.0
  assert holder.min_v==1.0


def test_all_equal_values() -> None:
  # values
  holder=MAMH()
  values=[5.0, 5.0, 5.0]

  # test
  for v in values:
    holder.append(v)

  # results
  assert holder.max_v==5.0
  assert holder.min_v==5.0
  assert holder.max_i==0
  assert holder.min_i==0
  assert holder.avg==5.0


def test_negative_values() -> None:
  # values
  holder=MAMH()
  values=[-10.0, -5.0, -20.0]

  # test
  for v in values:
    holder.append(v)

  # results
  assert holder.max_v==-5.0
  assert holder.min_v==-20.0
  assert holder.max_i==1
  assert holder.min_i==2
  assert holder.sum==sum(values)

def test_arr_returns_copy1() -> None:
  # values
  holder=MAMH()
  holder.append(1.0)
  holder.append(2.0)

  # test
  arr=holder.arr
  arr.append(999.0)

  # results
  assert holder.arr==[1.0, 2.0]
  assert arr==[1.0, 2.0, 999.0]

def test_arr_returns_copy2() -> None:
  # values
  holder=MAMH()
  holder.append(1.0)
  holder.append(2.0)

  # test
  arr=holder.arr
  holder.append(999.0)

  # results
  assert holder.arr==[1.0, 2.0, 999.0]
  assert arr==[1.0, 2.0]
