import pytest
from gaaml.classes import _utils as util

def test_center_of_range() -> None:
  # values
  p_s=(3, -1, 3, 2)
  k_s=(8, 6, 7, -5)
  e_s=(5, 2, 5, -1)

  for p, k, expected in zip(p_s, k_s, e_s):
    # test
    ret=util._center_of_range(p, k)

    # results
    assert isinstance(ret, int)
    assert ret==expected

def test_get_i_in_range() -> None:
  """
  In: l=[.3, .4, .6, .8, .9, 1] and v=.87
  Out: i=4
  In: l=[.3, .4, .6, .8, .9, 1] and v=.4
  Out: i=2
  In: l=[.3, .4, .6, .8, .9, .999] and v=1
  Out: i=5
  """
  # values
  l=[.3, .4, .6, .8, .9, 1]
  v_s=(0, .1, .4, .87, 1, 2)
  expected_s=(0, 0, 2, 4, 5, 5)

  for v, expected in zip(v_s, expected_s):
    # test
    ret=util.get_i_in_range(l, v)

    # results
    assert isinstance(ret, int)
    assert ret==expected

def test_randint() -> None:
  # values
  p=2
  k=5

  for _ in range(50): # random
    # test
    ret=util.randint(p, k)

    # results
    assert isinstance(ret, int)
    assert p<=ret
    assert ret<=k

def test_int_to_bin() -> None:
  """
  value=22
  bin: 1 0 1 1 0 (22d)
  """
  # values
  val=22
  length=5

  # test
  ret=util.int_to_bin(val, length)

  # results
  expected=bytearray('1 0 1 1 0'.replace(' ', '').encode())
  assert isinstance(ret, bytearray)
  assert len(ret)==length
  assert ret==expected

def test_correct_gen_to_min_max1() -> None:
  """
  gen: 1 1 1 0 1 (29d)
  max_v=27
  min_v=5

  max: 1 0 1 1 0 (22d)
  min: 0 0 0 0 0 (0d)

  ret: 1 0 1 0 1 (21d)
  ret=21+5=26
  """
  # values
  gen=bytearray('1 1 1 0 1'.replace(' ', '').encode())

  max_val=27
  min_val=5

  # test
  ret=util.correct_gen_to_min_max(gen, min_val, max_val)

  # results
  expected=int(bytearray('1 0 1 0 1'.replace(' ', '').encode()), 2)+min_val
  assert isinstance(ret, int)
  assert ret==expected

def test_correct_gen_to_min_max2() -> None:
  """
  gen: 1 0 1 1 1 (23d)
  max_v=27
  min_v=5

  max: 1 0 1 1 0 (22d)
  min: 0 0 0 0 0 (0d)

  ret: 1 0 1 1 0 (22d)
  ret=22+5=27
  """
  # values
  gen=bytearray('1 0 1 1 1'.replace(' ', '').encode())

  max_val=27
  min_val=5

  # test
  ret=util.correct_gen_to_min_max(gen, min_val, max_val)

  # results
  expected=int(bytearray('1 0 1 1 0'.replace(' ', '').encode()), 2)+min_val
  assert isinstance(ret, int)
  assert ret==expected

def test_correct_gen_to_min_max3() -> None:
  """
  gen: 1 1 0 1 1 (27d)
  max_v=26
  min_v=5

  max: 1 0 1 0 1 (21d)
  min: 0 0 0 0 0 (0d)

  ret: 1 0 0 1 1 (19d)
  ret=19+5=24
  """
  # values
  gen=bytearray('1 1 0 1 1'.replace(' ', '').encode())

  max_val=26
  min_val=5

  # test
  ret=util.correct_gen_to_min_max(gen, min_val, max_val)

  # results
  expected=int(bytearray('1 0 0 1 1'.replace(' ', '').encode()), 2)+min_val
  assert isinstance(ret, int)
  assert ret==expected

# def test_error_not_same_type_on_get_cp() -> None:
#   input_len1=5
#   input_len2=6
#   gi1=GI(input_len1)
#   gi2=GI(input_len2)
#   with pytest.raises(ValueError) as excinfo:
#     _=GI.get_cp(gi1, gi2)
#   assert str(excinfo.value)=='First and second solution are not equal in size'

# def test_error_illegal_argument_on_create() -> None:
#   input_len=5
#   gi1=GI(input_len)
#   gi2=GI(input_len)
#   for args, kwargs in (
#     ((gi1, None), {'cross_point': 2}),
#     ((gi1, gi2), {}),
#     ((gi1, gi2), {'cross_point': None}),
#     ((gi1, None), {}),
#     ((gi1, None), {'cross_point': None}),
#   ):
#     with pytest.raises(ValueError) as excinfo:
#       _=GI(*args, **kwargs) # type: ignore
#     assert str(excinfo.value)=='Illegal argument options'
