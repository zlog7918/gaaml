import pytest
from gaaml.classes.GenIndividual import GenIndividual as GI

def test_create() -> None:
  input_len=5
  gi=GI(input_len)
  assert isinstance(gi.gen, bytearray)
  assert len(gi.gen)==input_len

def test_mutate() -> None:
  input_len=5
  gi=GI(input_len)
  oryg_gen=gi.gen[:]
  gi.mutate()
  assert gi.gen!=oryg_gen

def test_get_cp() -> None:
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  cp=GI.get_cp(gi1, gi2)
  assert isinstance(cp, int)
  assert 0<=cp
  assert cp<=input_len

def test_create_from_two() -> None:
  """
  0 1 1|0 0
  1 0 0|0 1
  cp:3 ↑

  expected:
  0 1 1|0 1
  """
  input_len=5
  gi1=GI(input_len)
  gi1.gen=bytearray('01100'.encode())
  gi2=GI(input_len)
  gi2.gen=bytearray('10001'.encode())
  for cp, expected_str in zip(
    range(input_len+1),
    ['10001', '00001', '01001', '01101', '01101', '01100']
  ):
    gi=GI(gi1, gi2, cross_point=cp)
    assert gi.gen==bytearray(expected_str.encode())
    assert len(gi.gen)==input_len

def test_crossover() -> None:
  """
  0 1 1|0 0
  1 0 0|0 1
  cp:3 ↑

  expected:
  0 1 1|0 1
  1 0 0|0 0
  """
  input_len=5
  gi1=GI(input_len)
  gi1.gen=bytearray('01100'.encode())
  gi2=GI(input_len)
  gi2.gen=bytearray('10001'.encode())
  for cp, expected_strs in zip(
    range(input_len+1),
    [('10001', '01100'), ('00001', '11100'), ('01001', '10100'), ('01101', '10000'), ('01101', '10000'), ('01100', '10001')]
  ):
    gi_s=GI.crossover(gi1, gi2, cp)
    assert isinstance(gi_s, tuple)
    assert isinstance(gi_s[0], GI)
    assert gi_s[0].gen==bytearray(expected_strs[0].encode())
    assert isinstance(gi_s[1], GI)
    assert gi_s[1].gen==bytearray(expected_strs[1].encode())

def test_error_not_same_type_on_get_cp() -> None:
  input_len1=5
  input_len2=6
  gi1=GI(input_len1)
  gi2=GI(input_len2)
  with pytest.raises(ValueError) as excinfo:
    _=GI.get_cp(gi1, gi2)
  assert str(excinfo.value)=='First and second solution are not equal in size'

def test_error_not_same_type_on_crossover() -> None:
  input_len1=5
  input_len2=6
  gi1=GI(input_len1)
  gi2=GI(input_len2)
  with pytest.raises(ValueError) as excinfo:
    _=GI.crossover(gi1, gi2, 2)
  assert str(excinfo.value)=='First and second solution are not equal in size'

def test_error_not_same_type_on_create() -> None:
  input_len1=5
  input_len2=6
  gi1=GI(input_len1)
  gi2=GI(input_len2)
  with pytest.raises(ValueError) as excinfo:
    _=GI(gi1, gi2, cross_point=2)
  assert str(excinfo.value)=='First and second solution are not equal in size'

def test_error_outside_range_on_crossover() -> None:
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  with pytest.raises(ValueError) as excinfo:
    _=GI.crossover(gi1, gi2, -1)
  assert str(excinfo.value)=='Cross point is outside of solution'
  with pytest.raises(ValueError) as excinfo:
    _=GI.crossover(gi1, gi2, 6)
  assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_outside_range_on_create() -> None:
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  with pytest.raises(ValueError) as excinfo:
    _=GI(gi1, gi2, cross_point=-1)
  assert str(excinfo.value)=='Cross point is outside of solution'
  with pytest.raises(ValueError) as excinfo:
    _=GI(gi1, gi2, cross_point=6)
  assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_illegal_argument_on_create() -> None:
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  for args, kwargs in (
    ((gi1, None), {'cross_point': 2}),
    ((gi1, gi2), {}),
    ((gi1, gi2), {'cross_point': None}),
    ((gi1, None), {}),
    ((gi1, None), {'cross_point': None}),
  ):
    with pytest.raises(ValueError) as excinfo:
      _=GI(*args, **kwargs) # type: ignore
    assert str(excinfo.value)=='Illegal argument options'
