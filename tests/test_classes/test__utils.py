import pytest
from gaaml.classes import _utils as util

mark__test_create_from_two=pytest.mark.parametrize(
  ('p', 'k', 'exp'),
  [
    *zip(
      (3, -1, 3, 2),
      (8, 6, 7, -5),
      (5, 2, 5, -1),
    )
  ],
)
@mark__test_create_from_two
def test_create_from_two(p: int, k: int, exp: int) -> None:
  # values ^

  # test
  ret=util._center_of_range(p, k)

  # results
  assert isinstance(ret, int)
  assert ret==exp

mark__test_get_i_in_range=pytest.mark.parametrize(
  ('v', 'exp'),
  [
    *zip(
      (0, .1, .4, .87, 1, 2),
      (0, 0, 2, 4, 5, 5),
    )
  ],
)
@mark__test_get_i_in_range
def test_get_i_in_range(v: float, exp: int) -> None:
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

  # test
  ret=util.get_i_in_range(l, v)

  # results
  assert isinstance(ret, int)
  assert ret==exp

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

mark__test_int_to_bin=pytest.mark.parametrize(
  ('val', 'length', 'exp_gen_str'),
  [
    (22, 6, '0 1 0 1 1 0'),
    (22, 5, '1 0 1 1 0'),
    (22, 4, '0 1 1 0'), # 1 0 1 1 0
  ],
)
@mark__test_int_to_bin
def test_int_to_bin(val: int, length: int, exp_gen_str: str) -> None:
  """
  value=22
  bin: 1 0 1 1 0 (22d)
  """
  # values ^

  # test
  ret=util.int_to_bin(val, length)

  # results
  expected=bytearray(exp_gen_str.replace(' ', '').encode())
  assert isinstance(ret, bytearray)
  assert len(ret)==length
  assert ret==expected

mark__test_correct_gen_to_min_max=pytest.mark.parametrize(
  ('gen_str', 'max_v', 'min_v', 'exp_val_in_str'),
  [
    ('1 1 1 0 1', 27, 5, '1 0 1 0 1'),
    ('1 0 1 1 1', 27, 5, '1 0 1 1 0'),
    ('1 1 0 1 1', 26, 5, '1 0 0 1 1'),
  ],
)
@mark__test_correct_gen_to_min_max
def test_correct_gen_to_min_max(gen_str: str, max_v: int, min_v: int, exp_val_in_str: str) -> None:
  """
  case 0:
    gen: 1 1 1 0 1 (29d)
    max_v=27
    min_v=5

    max: 1 0 1 1 0 (22d)
    min: 0 0 0 0 0 (0d)

    ret: 1 0 1 0 1 (21d)
    ret=21+5=26

  case 1:
    gen: 1 0 1 1 1 (23d)
    max_v=27
    min_v=5

    max: 1 0 1 1 0 (22d)
    min: 0 0 0 0 0 (0d)

    ret: 1 0 1 1 0 (22d)
    ret=22+5=27

  case 2:
    gen: 1 1 0 1 1 (27d)
    max_v=26
    min_v=5

    max: 1 0 1 0 1 (21d)
    min: 0 0 0 0 0 (0d)

    ret: 1 0 0 1 1 (19d)
    ret=19+5=24
  """
  # values
  gen=bytearray(gen_str.replace(' ', '').encode())

  # test
  ret=util.correct_gen_to_min_max(gen, min_v, max_v)

  # results
  expected=int(bytearray(exp_val_in_str.replace(' ', '').encode()), 2)+min_v
  assert isinstance(ret, int)
  assert ret==expected

mark__test_error_int_to_bin=pytest.mark.parametrize(
  ('val', 'length'),
  [
    (22, -6),
    (22, -1),
    (22, 0),
  ],
)
@mark__test_error_int_to_bin
def test_error_int_to_bin(val: int, length: int) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=util.int_to_bin(val, length)

  # results
  assert str(excinfo.value)=='length must be greater then 0'
