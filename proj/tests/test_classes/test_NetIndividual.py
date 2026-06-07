import json
import pytest
import typing as t
from gaaml.classes import _utils as util
from gaaml.classes.NetIndividual import NetIndividual as NI
from gaaml.classes.MaxIntsIndividual import MaxIntsIndividual as _MII
from gaaml.classes.MaxIntsListIndividual import MaxIntsListIndividual as _MILI

mark__test_type_len_modifier=pytest.mark.parametrize(
  ('l', 'exp_l'),
  [
    (4, 5),
    (7, 8),
    (16, 17),
    (1, 2),
  ]
)
@mark__test_type_len_modifier
def test_type_len_modifier(l: int, exp_l: int) -> None:
  # values ^

  # test
  ret_l=NI.type_len_modifier(l)

  # results
  assert ret_l==exp_l

def test_create() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  exp_len_name, (exp_len_bit, exp_len_min, exp_len_max)=layers_len
  exp_num_seed_name, (exp_num_seed_bit, exp_num_seed_min, exp_num_seed_max)=num_seed
  exp_type_seed_name, (exp_type_seed_bit, exp_type_seed_min, exp_type_seed_max)=type_seed
  (
    (exp_x_name, (exp_x_bit, exp_x_min, exp_x_max)),
    (exp_y_name, (exp_y_bit, exp_y_min, exp_y_max)),
  )=g_schema
  exp_n_bit, exp_n_min, exp_n_max=n_schema
  exp_t_bit, exp_t_min, exp_t_max=t_schema
  expected_0gen=exp_len_bit+exp_num_seed_bit+exp_type_seed_bit+exp_x_bit+exp_y_bit

  for _ in range(50): # random process
    # test
    ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

    # results
    assert isinstance(ni.gen, tuple)
    assert ni.layers_len_name==exp_len_name
    assert ni.num_seed_name==exp_num_seed_name
    assert ni.type_seed_name==exp_type_seed_name
    assert len(ni.gen)==3
    mii, mili_n, mili_t=ni.gen

    assert isinstance(mii, _MII)
    assert isinstance(mii.gen.gen, bytearray)
    assert len(mii.gen.gen)==expected_0gen
    assert isinstance(mii.fenotype, dict)
    assert {*mii.fenotype.keys()}=={exp_len_name, exp_num_seed_name, exp_type_seed_name, exp_x_name, exp_y_name}
    _sum=0
    for name, bits, _min, _max in (
      (exp_len_name, exp_len_bit, exp_len_min, exp_len_max),
      (exp_num_seed_name, exp_num_seed_bit, exp_num_seed_min, exp_num_seed_max),
      (exp_type_seed_name, exp_type_seed_bit, exp_type_seed_min, exp_type_seed_max),
      (exp_x_name, exp_x_bit, exp_x_min, exp_x_max),
      (exp_y_name, exp_y_bit, exp_y_min, exp_y_max),
    ):
      prev=_sum
      _sum+=bits
      assert mii.fenotype[name]==util.correct_gen_to_min_max(mii.gen.gen[prev:_sum], _min, _max)

    expected_num_len=mii.fenotype[exp_len_name]
    expected_type_len=NI.type_len_modifier(expected_num_len)
    assert isinstance(mili_n, _MILI)
    assert isinstance(mili_n.fenotype, list)
    assert len(mili_n.gen.gen)%exp_n_bit==0
    assert len(mili_n.gen.gen)>=expected_num_len*exp_n_bit
    assert len(mili_n.fenotype)==expected_num_len
    prev_i=0
    for i, bit_i in zip(
      range(expected_num_len),
      range(exp_n_bit, expected_num_len*exp_n_bit+1, exp_n_bit),
    ):
      assert mili_n.fenotype[i]==util.correct_gen_to_min_max(mili_n.gen.gen[prev_i:bit_i], exp_n_min, exp_n_max)
      prev_i=bit_i

    assert isinstance(mili_t, _MILI)
    assert isinstance(mili_t.fenotype, list)
    assert len(mili_t.gen.gen)%exp_t_bit==0
    assert len(mili_t.gen.gen)>=expected_type_len*exp_t_bit
    assert len(mili_t.fenotype)==expected_type_len
    prev_i=0
    for i, bit_i in zip(
      range(expected_type_len),
      range(exp_t_bit, expected_type_len*exp_t_bit+1, exp_t_bit),
    ):
      assert mili_t.fenotype[i]==util.correct_gen_to_min_max(mili_t.gen.gen[prev_i:bit_i], exp_t_min, exp_t_max)
      prev_i=bit_i

def test_mutate() -> None:
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  for _ in range(50): # random process
    oryg_gen=(
      ni.gen[0].gen.gen[:],
      ni.gen[1].gen.gen[:],
      ni.gen[2].gen.gen[:],
    )

    # test
    ni.mutate()

    # results
    assert (
      ni.gen[0].gen.gen,
      ni.gen[1].gen.gen,
      ni.gen[2].gen.gen,
    )!=oryg_gen

def test_get_cp() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni1=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni2=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  len_name, (len_bit, len_min, _)=layers_len
  bit_len1=ni1.gen[0].fenotype[len_name]
  bit_len2=ni2.gen[0].fenotype[len_name]

  _, (num_seed_bit, _, _)=num_seed
  _, (type_seed_bit, _, _)=type_seed
  (
    (_, (x_bit, _, _)),
    (_, (y_bit, _, _)),
  )=g_schema
  n_bit, _, _=n_schema
  t_bit, _, _=t_schema
  gen0_bit=len_bit+num_seed_bit+type_seed_bit+x_bit+y_bit

  for _ in range(50): # random process
    # test
    cp=NI.get_cp(ni1, ni2)

    # results
    assert isinstance(cp, tuple)
    assert isinstance(cp[1], tuple)
    assert isinstance(cp[2], tuple)
    cp0, (cp11, cp12), (cp21, cp22)=cp
    assert isinstance(cp0, int)
    assert 0<=cp0
    assert cp0<=gen0_bit

    assert isinstance(cp11, int)
    assert isinstance(cp12, int)
    assert 0<=cp11
    assert 0<=cp12
    assert cp11<=bit_len1*n_bit
    assert cp12<=bit_len2*n_bit
    assert cp11%n_bit==cp12%n_bit
    assert cp11+(bit_len2*n_bit-cp12)>=len_min*n_bit
    assert (bit_len1*n_bit-cp11)+cp12>=len_min*n_bit

    bit_len1, bit_len2=NI.type_len_modifier(bit_len1), NI.type_len_modifier(bit_len2)
    assert isinstance(cp21, int)
    assert isinstance(cp22, int)
    assert 0<=cp21
    assert 0<=cp22
    assert cp21<=bit_len1*t_bit
    assert cp22<=bit_len2*t_bit
    assert cp21%t_bit==cp22%t_bit
    assert cp21+(bit_len2*t_bit-cp22)>=len_min*t_bit
    assert (bit_len1*t_bit-cp21)+cp22>=len_min*t_bit

def test_create_from_two() -> None:
  """
          len  n_s  t_s   x     y
  bits01: 0 0  1 0  1 1  1 0  1 0 1
  bits02: 0 1  0 1  0 0  0 0  0 1 1
  cp0:1    ↑

  bits_n1: 1 1 0
              ↑
  bits_n2: 0 1 0  0 0 1
                     ↑
  cp1:2,5

  bits_t1: 1 1  0 1
                 ↑
  bits_t2: 1 0  0 1  0 0
            ↑
  cp1:3,1

  expected:
  bits0: 0 1  0 1  0 0  0 0  0 1 1
  bits_n: 1 1 1
  bits_t: 1 1  0 0  0 1  0 0
  """
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni1=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni2=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema

  # setup
  ni1.gen[0]._gen._gen=bytearray('0 0  1 0  1 1  1 0  1 0 1'.replace(' ', '').encode())
  ni1.gen[1]._gen._gen=bytearray('1 1 0'.replace(' ', '').encode())
  ni1.gen[2]._gen._gen=bytearray('1 1  0 1'.replace(' ', '').encode())
  ni2.gen[0]._gen._gen=bytearray('0 1  0 1  0 0  0 0  0 1 1'.replace(' ', '').encode())
  ni2.gen[1]._gen._gen=bytearray('0 1 0  0 0 1'.replace(' ', '').encode())
  ni2.gen[2]._gen._gen=bytearray('1 0  0 1  0 0'.replace(' ', '').encode())

  for cp, (exp_str0, exp_str1, exp_str2), (exp_feno0, exp_feno1, exp_feno2) in (
    (
      (1, (2, 5), (3, 1)), (
        '0 1  0 1  0 0  0 0  0 1 1',
        '1 1 1',
        '1 1  0 0  0 1  0 0',
      ), (
        {len_name: 2, num_seed_name: 1, type_seed_name: 0, x_name: 1, y_name: 3},
        [7, 3], # one more: len->2, filled in by random with num_seed(1)->3
        [3, 0, 1], # one less: len->2 -> 2+2=4
      )
    ),
  ):
    # test
    ni=NI(ni1, ni2, cross_point=cp)

    # results
    mii, mili_n, mili_t=ni.gen
    assert isinstance(mii, _MII)
    assert isinstance(mii.gen.gen, bytearray)
    assert mii.gen.gen==bytearray(exp_str0.replace(' ', '').encode())
    assert isinstance(mii.fenotype, dict)
    assert mii.fenotype==exp_feno0

    assert isinstance(mili_n, _MILI)
    assert isinstance(mili_n.gen.gen, bytearray)
    assert mili_n.gen.gen==bytearray(exp_str1.replace(' ', '').encode())
    assert isinstance(mili_n.fenotype, list)
    assert mili_n.fenotype==exp_feno1

    assert isinstance(mili_t, _MILI)
    assert isinstance(mili_t.gen.gen, bytearray)
    assert mili_t.gen.gen==bytearray(exp_str2.replace(' ', '').encode())
    assert isinstance(mili_t.fenotype, list)
    assert mili_t.fenotype==exp_feno2

def test_crossover() -> None:
  """
          len  n_s  t_s   x     y
  bits01: 0 0  1 0  1 1  1 0  1 0 1
  bits02: 0 1  0 1  0 0  0 0  0 1 1
  cp0:1    ↑

  bits_n1: 1 1 0
              ↑
  bits_n2: 0 1 0  0 0 1
                     ↑
  cp1:2,5

  bits_t1: 1 1  0 1
            ↑
  bits_t2: 1 0  0 1  0 0
                 ↑
  cp1:1,3

  expected:
  child1:
  bits0: 0 1  0 1  0 0  0 0  0 1 1
  bits_n: 1 1 1
  bits_t: 1 1  0 0
  child2:
  bits0: 0 0  1 0  1 1  1 0  1 0 1
  bits_n: 0 1 0  0 0 0
  bits_t: 1 0  0 1  0 1
  """
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni1=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni2=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema

  # setup
  ni1.gen[0]._gen._gen=bytearray('0 0  1 0  1 1  1 0  1 0 1'.replace(' ', '').encode())
  ni1.gen[1]._gen._gen=bytearray('1 1 0'.replace(' ', '').encode())
  ni1.gen[2]._gen._gen=bytearray('1 1  0 1'.replace(' ', '').encode())
  ni2.gen[0]._gen._gen=bytearray('0 1  0 1  0 0  0 0  0 1 1'.replace(' ', '').encode())
  ni2.gen[1]._gen._gen=bytearray('0 1 0  0 0 1'.replace(' ', '').encode())
  ni2.gen[2]._gen._gen=bytearray('1 0  0 1  0 0'.replace(' ', '').encode())

  for (
    cp,
    (
      (exp_str01, exp_str11, exp_str21),
      (exp_feno01, exp_feno11, exp_feno21),
    ),
    (
      (exp_str02, exp_str12, exp_str22),
      (exp_feno02, exp_feno12, exp_feno22),
    )
  ) in (
    (
      (1, (2, 5), (1, 3)), (
        (
          '0 1  0 1  0 0  0 0  0 1 1',
          '1 1 1',
          '1 1  0 0',
        ),
        (
          {len_name: 2, num_seed_name: 1, type_seed_name: 0, x_name: 1, y_name: 3},
          [7, 3], # one more: len->2, filled in by random with num_seed(1)->3
          [3, 0, 3], # one more: len->2, filled in by random with type_seed(0)->3
        ),
      ), (
        (
          '0 0  1 0  1 1  1 0  1 0 1',
          '0 1 0  0 0 0',
          '1 0  0 1  0 1',
        ),
        (
          {len_name: 1, num_seed_name: 2, type_seed_name: 3, x_name: 3, y_name: 5},
          [4], # one less: len->1
          [2, 1], # one less: len->1 -> 1+1=2
        ),
      )
    ),
  ):
    # test
    ni_s=NI.crossover(ni1, ni2, cp)

    # results
    for ni, (exp_str0, exp_str1, exp_str2), (exp_feno0, exp_feno1, exp_feno2) in zip(
      ni_s,
      ((exp_str01, exp_str11, exp_str21), (exp_str02, exp_str12, exp_str22)),
      ((exp_feno01, exp_feno11, exp_feno21), (exp_feno02, exp_feno12, exp_feno22)),
    ):
      mii, mili_n, mili_t=ni.gen
      assert isinstance(mii, _MII)
      assert isinstance(mii.gen.gen, bytearray)
      assert mii.gen.gen==bytearray(exp_str0.replace(' ', '').encode())
      assert isinstance(mii.fenotype, dict)
      assert mii.fenotype==exp_feno0

      assert isinstance(mili_n, _MILI)
      assert isinstance(mili_n.gen.gen, bytearray)
      assert mili_n.gen.gen==bytearray(exp_str1.replace(' ', '').encode())
      assert isinstance(mili_n.fenotype, list)
      assert mili_n.fenotype==exp_feno1

      assert isinstance(mili_t, _MILI)
      assert isinstance(mili_t.gen.gen, bytearray)
      assert mili_t.gen.gen==bytearray(exp_str2.replace(' ', '').encode())
      assert isinstance(mili_t.fenotype, list)
      assert mili_t.fenotype==exp_feno2

def test_update_fenotype() -> None:
  """
         len  n_s  t_s   x     y
  bits0: 0 1  1 0  1 1  1 0  1 1 0

  bits_n: 0 1 0  0 0 1
  bits_t: 1 0  0 1  0 0

  expected:
  fenotype0: {
    len: 1+1=2
    n_s: 2+0
    t_s: 2+0
    x: 2+1=3
    y: 6+0=6 -> 6->4 -> 4+0=4
  }
  fenotype1: [2+2->4, 1+2->3]
  fenotype2: [2+0->2, 1+0->1, 0+0->0]
  """
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema

  for (str0, str1, str2), (exp_feno0, (exp_feno1, feno_corr1), (exp_feno2, feno_corr2)) in (
    (
      (
        '0 1  0 1  0 0  0 0  0 1 1',
        '1 1 1',
        '1 0  1 1  0 0  0 1  0 0',
      ), (
        {len_name: 2, num_seed_name: 1, type_seed_name: 0, x_name: 1, y_name: 3},
        ([7, 3], True), # one more: len->2, filled in by random with num_seed(1)->3
        ([2, 3, 0], True), # one less: len->2 -> 2+1=3
      )
    ),
    (
      (
        '1 1  0 1  0 1  0 1  0 1 1',
        '1 1 1  1 0 1  0 1 1  0 0 0',
        '1 0  1 1',
      ), (
        {len_name: 4, num_seed_name: 1, type_seed_name: 1, x_name: 2, y_name: 3},
        ([7, 7, 5, 2], False),
        ([2, 3, 1, 0, 2], True), # three more: len->4 -> 4+1=5, filled in by random with num_seed(1)->[1, 0, 2]
      )
    ),
  ):
    # setup
    for i, _str in zip(range(3), (str0, str1, str2)):
      ni.gen[i]._gen._gen=bytearray(_str.replace(' ', '').encode())
      ni.gen[i]._update_fenotype()

    oryg_fenotype0={k: v for k,v in ni.gen[0].fenotype.items()}
    oryg_fenotype1=ni.gen[1].fenotype[:]
    oryg_fenotype2=ni.gen[2].fenotype[:]

    # test
    ni._update()

    # results
    mii, mili_n, mili_t=ni.gen
    assert mii.gen.gen==bytearray(str0.replace(' ', '').encode())
    assert isinstance(mii.fenotype, dict)
    assert mii.fenotype==oryg_fenotype0
    assert mii.fenotype==exp_feno0

    assert mili_n.gen.gen==bytearray(str1.replace(' ', '').encode())
    assert isinstance(mili_n.fenotype, list)
    assert (mili_n.fenotype!=oryg_fenotype1)==feno_corr1
    assert mili_n.fenotype==exp_feno1

    assert mili_t.gen.gen==bytearray(str2.replace(' ', '').encode())
    assert isinstance(mili_t.fenotype, list)
    assert (mili_t.fenotype!=oryg_fenotype2)==feno_corr2
    assert mili_t.fenotype==exp_feno2

def test_save_format() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  gen_g, gen_l, gen_t=ni.gen

  # test
  ret=ni._save_format()

  # results
  assert ret=={
    'name': NI.__name__,
    'gen': {
      'g': gen_g._save_format(),
      'l': gen_l._save_format(),
      't': gen_t._save_format(),
    },
    'layers_len_name': ni.layers_len_name,
    'num_seed_name': ni.num_seed_name,
    'type_seed_name': ni.type_seed_name,
  }

def test_save_format_returns_serializable_data():
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  result=ni._save_format()

  # test/results
  _=json.dumps(result)

def test_load_from_format() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  l_schema=3, 2, 7
  t_schema=2, 0, 3
  layers_len_name, (_, len_min, len_max)=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  g_gen=_MII((layers_len, num_seed, type_seed, *g_schema))
  list_len=g_gen.fenotype[layers_len_name]
  l_gen=_MILI(list_len, ((len_min, len_max+1), l_schema))
  t_gen=_MILI(list_len+1, ((len_min+1, len_max+2), t_schema))

  saved={
    'name': NI.__name__,
    'gen': {
      'g': g_gen._save_format(),
      'l': l_gen._save_format(),
      't': t_gen._save_format(),
    },
    'layers_len_name': layers_len_name,
    'num_seed_name': num_seed_name,
    'type_seed_name': type_seed_name,
  }

  # test
  loaded=NI._load_from_format(saved)

  # results
  assert isinstance(loaded, NI)
  assert loaded.gen[0].gen.gen==g_gen.gen.gen
  assert loaded.gen[1].gen.gen==l_gen.gen.gen
  assert loaded.gen[2].gen.gen==t_gen.gen.gen

  assert loaded.layers_len_name==layers_len_name
  assert loaded.num_seed_name==num_seed_name
  assert loaded.type_seed_name==type_seed_name

def test_load_from_format_roundtrip() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  ni=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # setup
  saved=json.loads(json.dumps(ni._save_format()))

  # test
  loaded=NI._load_from_format(saved)

  # results
  assert isinstance(loaded, NI)
  assert loaded.gen[0].gen.gen==ni.gen[0].gen.gen
  assert loaded.gen[1].gen.gen==ni.gen[1].gen.gen
  assert loaded.gen[2].gen.gen==ni.gen[2].gen.gen

def test_error_name_collition_on_create1() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('y', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create2() -> None:
  # values
  layers_len=('x', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create3() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('len', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create4() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('x', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create5() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('len', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create6() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('num_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

def test_error_name_collition_on_create7() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('x', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))

  # results
  assert str(excinfo.value)=='Names can not collide'

mark__test_error_not_same_conf_on_get_cp=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_conf_on_get_cp
def test_error_not_same_conf_on_get_cp(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.get_cp(ni1, ni2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_not_same_gen_size_on_get_cp=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_gen_size_on_get_cp
def test_error_not_same_gen_size_on_get_cp(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.get_cp(ni1, ni2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

mark__test_error_not_same_conf_on_create=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_conf_on_create
def test_error_not_same_conf_on_create(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_not_same_gen_size_on_create=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_gen_size_on_create
def test_error_not_same_gen_size_on_create(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  dummy_cp=1, (1, 1), (1, 1)

  ni2=NI(*schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

mark__test_error_not_same_conf_on_crossover=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_conf_on_crossover
def test_error_not_same_conf_on_crossover(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_not_same_gen_size_on_crossover=pytest.mark.parametrize(
  'calc_schema2',
  [
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # lambda layers_len, num_seed, type_seed, g_schema, n_schema, t_schema: (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ]
)
@mark__test_error_not_same_gen_size_on_crossover
def test_error_not_same_gen_size_on_crossover(
  calc_schema2: t.Callable[
    [
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], ...],
      tuple[int, int, int],
      tuple[int, int, int],
    ],
    tuple[
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      NI.GenSchemaType,
    ],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  schema2=calc_schema2(layers_len, num_seed, type_seed, g_schema, n_schema, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

mark__test_error_outside_range_on_create=pytest.mark.parametrize(
  'cp_calc_func',
  [
    lambda len1, len2, gen0_bit, n_bit, t_bit: (-1, (1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (gen0_bit+1, (1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (-1, -1%n_bit), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (-1%n_bit, -1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (-1, -1%t_bit)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (-1%t_bit, -1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (len1*n_bit+1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, len2*n_bit+1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (NI.type_len_modifier(len1)*t_bit+1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (1, NI.type_len_modifier(len2)*t_bit+1)),
    # lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (1, 1)),
  ]
)
@mark__test_error_outside_range_on_create
def test_error_outside_range_on_create(
  cp_calc_func: t.Callable[
    [int, int, int, int, int],
    tuple[int, tuple[int, int], tuple[int, int]],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  ni1=NI(*schema)
  ni2=NI(*schema)

  len_name, (len_bit, _, _)=layers_len
  len1=ni1.gen[0].fenotype[len_name]
  len2=ni2.gen[0].fenotype[len_name]
  _, (num_seed_bit, _, _)=num_seed
  _, (type_seed_bit, _, _)=type_seed
  (
    (_, (x_bit, _, _)),
    (_, (y_bit, _, _)),
  )=g_schema
  n_bit, _, _=n_schema
  t_bit, _, _=t_schema
  gen0_bit=len_bit+num_seed_bit+type_seed_bit+x_bit+y_bit
  cp=cp_calc_func(len1, len2, gen0_bit, n_bit, t_bit)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_outside_range_on_crossover=pytest.mark.parametrize(
  'cp_calc_func',
  [
    lambda len1, len2, gen0_bit, n_bit, t_bit: (-1, (1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (gen0_bit+1, (1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (-1, -1%n_bit), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (-1%n_bit, -1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (-1, -1%t_bit)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (-1%t_bit, -1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (len1*n_bit+1, 1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, len2*n_bit+1), (1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (NI.type_len_modifier(len1)*t_bit+1, 1)),
    lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (1, NI.type_len_modifier(len2)*t_bit+1)),
    # lambda len1, len2, gen0_bit, n_bit, t_bit: (1, (1, 1), (1, 1)),
  ]
)
@mark__test_error_outside_range_on_crossover
def test_error_outside_range_on_crossover(
  cp_calc_func: t.Callable[
    [int, int, int, int, int],
    tuple[int, tuple[int, int], tuple[int, int]],
  ]
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  ni1=NI(*schema)
  ni2=NI(*schema)

  len_name, (len_bit, _, _)=layers_len
  len1=ni1.gen[0].fenotype[len_name]
  len2=ni2.gen[0].fenotype[len_name]
  _, (num_seed_bit, _, _)=num_seed
  _, (type_seed_bit, _, _)=type_seed
  (
    (_, (x_bit, _, _)),
    (_, (y_bit, _, _)),
  )=g_schema
  n_bit, _, _=n_schema
  t_bit, _, _=t_schema
  gen0_bit=len_bit+num_seed_bit+type_seed_bit+x_bit+y_bit
  cp=cp_calc_func(len1, len2, gen0_bit, n_bit, t_bit)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_not_same_elem_size_on_create1() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema1=3, 2, 7
  n_schema2=4, 2, 7
  t_schema=2, 0, 3
  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema1, t_schema)
  schema2=layers_len, num_seed, type_seed, (g_schema, n_schema2, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_elem_size_on_create2() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema1=2, 0, 3
  t_schema2=3, 0, 3
  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema1)
  schema2=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema2)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_elem_size_on_crossover1() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema1=3, 2, 7
  n_schema2=4, 2, 7
  t_schema=2, 0, 3
  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema1, t_schema)
  schema2=layers_len, num_seed, type_seed, (g_schema, n_schema2, t_schema)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_elem_size_on_crossover2() -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema1=2, 0, 3
  t_schema2=3, 0, 3
  schema1=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema1)
  schema2=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema2)
  ni1=NI(*schema1)
  ni2=NI(*schema2)
  dummy_cp=1, (1, 1), (1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

mark__test_error_too_short_on_create_from_two=pytest.mark.parametrize(
  'cp',
  [
    (1, (0, 3), (1, 1)),
    (1, (1, 4), (1, 1)),
    (1, (2, 5), (1, 1)),
    (1, (3, 6), (1, 1)),
    (1, (1, 1), (4, 8)),
    (1, (1, 1), (3, 7)),
    (1, (1, 1), (1, 7)),
    (1, (1, 1), (1, 5)),
    # (1, (1, 1), (1, 1)),
  ]
)
@mark__test_error_too_short_on_create_from_two
def test_error_too_short_on_create_from_two(cp: tuple[int, tuple[int, int], tuple[int, int]]) -> None:
  # values
  layers_len=('len', (2, 2, 5))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  schema=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  ni1=NI(*schema)
  ni2=NI(*schema)

  # setup
  for ni, (str0, str1, str2) in (
    (
      ni1,
      (
        '0 0  0 1  0 0  0 0  0 1 1',
        '1 1 1  0 1 1',
        '1 0  1 1  0 1  0 0',
      ),
    ),
    (
      ni2,
      (
        '1 1  0 1  0 1  0 1  0 1 1',
        '1 0 1  0 0 0',
        '1 0  1 1  0 0  0 1',
      ),
    ),
  ):
    for i, _str in zip(range(3), (str0, str1, str2)):
      ni.gen[i]._gen._gen=bytearray(_str.replace(' ', '').encode())
      ni.gen[i]._update_fenotype()
    ni._update()

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(ni1, ni2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Solution too short'

mark__test_error_too_short_on_crossover=pytest.mark.parametrize(
  'cp',
  [
    # first too short
    (1, (0, 3), (1, 1)),
    (1, (1, 4), (1, 1)),
    (1, (2, 5), (1, 1)),
    (1, (3, 6), (1, 1)),
    (1, (1, 1), (4, 8)),
    (1, (1, 1), (3, 7)),
    (1, (1, 1), (1, 7)),
    (1, (1, 1), (1, 5)),
    # second too short
    (1, (3, 0), (1, 1)),
    (1, (4, 1), (1, 1)),
    (1, (5, 2), (1, 1)),
    (1, (6, 3), (1, 1)),
    (1, (1, 1), (8, 4)),
    (1, (1, 1), (7, 3)),
    (1, (1, 1), (7, 1)),
    (1, (1, 1), (5, 1)),
  ]
)
@mark__test_error_too_short_on_crossover
def test_error_too_short_on_crossover(cp: tuple[int, tuple[int, int], tuple[int, int]]) -> None:
  # values
  layers_len=('len', (2, 2, 5))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  schema=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  ni1=NI(*schema)
  ni2=NI(*schema)

  # setup
  for ni, (str0, str1, str2) in (
    (
      ni1,
      (
        '0 0  0 1  0 0  0 0  0 1 1',
        '1 1 1  0 1 1',
        '1 0  1 1  0 1  0 0',
      ),
    ),
    (
      ni2,
      (
        '1 1  0 1  0 1  0 1  0 1 1',
        '1 0 1  0 0 0',
        '1 0  1 1  0 0  0 1',
      ),
    ),
  ):
    for i, _str in zip(range(3), (str0, str1, str2)):
      ni.gen[i]._gen._gen=bytearray(_str.replace(' ', '').encode())
      ni.gen[i]._update_fenotype()
    ni._update()

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI.crossover(ni1, ni2, cp)

  # results
  assert str(excinfo.value)=='Solution too short'

mark__test_error_load_from_format=pytest.mark.parametrize(
  'saved_model',
  [

    # missing / invalid keys
    {},
    {'name': NI.__name__},
    {'gen': {}},
    {'gen': {}, 'name': NI.__name__},
    {'gen': {}, 'name': NI.__name__, 'extra': 123},

    # wrong name
    {'name': 'OtherClass', 'gen': {}, 'layers_len_name': 'a', 'num_seed_name': 'b', 'type_seed_name': 'c'},
    {'name': '_NI', 'gen': {}, 'layers_len_name': 'a', 'num_seed_name': 'b', 'type_seed_name': 'c'},

    # missing required scalar fields
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      # missing layers_len_name
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      'layers_len_name': 'len',
      # missing num_seed_name
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      # missing type_seed_name
    },

    # wrong types (scalar fields)
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      'layers_len_name': 123,
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      'layers_len_name': 'len',
      'num_seed_name': 123,
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}, 't': {}},
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 123,
    },

    # gen wrong type
    {
      'name': NI.__name__,
      'gen': [],
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': 'not-a-dict',
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },

    # gen missing keys
    {
      'name': NI.__name__,
      'gen': {'g': {}, 'l': {}},  # missing t
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'g': {}, 't': {}},  # missing l
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
    {
      'name': NI.__name__,
      'gen': {'l': {}, 't': {}},  # missing g
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },

    # nested load failures (invalid inner gens)
    {
      'name': NI.__name__,
      'gen': {
        'g': {'invalid': True},
        'l': {'invalid': True},
        't': {'invalid': True},
      },
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },

    # partially invalid nested gens
    {
      'name': NI.__name__,
      'gen': {
        'g': {'invalid': True},
        'l': _MILI(2, ((1, 4), (3, 1, 6)))._save_format(),
        't': _MILI(2, ((1, 4), (3, 1, 6)))._save_format(),
      },
      'layers_len_name': 'len',
      'num_seed_name': 'num',
      'type_seed_name': 'type',
    },
  ],
)
@mark__test_error_load_from_format
def test_error_load_from_format(saved_model: dict[str, object]) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI._load_from_format(saved_model)

  assert str(excinfo.value)==f'Model saved is not {NI.__name__}'

mark__test_error_illegal_argument_on_create=pytest.mark.parametrize(
  'func_args_kwargs',
  [
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, None), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, None), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, num_seed, type_seed, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, None), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, ni2), {}),
    lambda ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp: ((ni1, ni2), {'cross_point': None}),
  ],
)
@mark__test_error_illegal_argument_on_create
def test_error_illegal_argument_on_create(
  func_args_kwargs: t.Callable[
    [
      NI,
      NI,
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[str, tuple[int, int, int]],
      tuple[tuple[str, tuple[int, int, int]], tuple[str, tuple[int, int, int]]],
      tuple[int, int, int],
      tuple[int, int, int],
      tuple[int, tuple[int, int], tuple[int, int]],
    ],
    tuple[tuple[t.Any, ...], dict[str, t.Any]]
  ],
) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3

  schema=layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)
  ni1=NI(*schema)
  ni2=NI(*schema)
  dummy_cp=(1, (1, 1), (1, 1))
  args, kwargs=func_args_kwargs(ni1, ni2, layers_len, num_seed, type_seed, g_schema, n_schema, t_schema, dummy_cp)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=NI(*args, **kwargs)

  # results
  assert str(excinfo.value)=='Illegal argument options'
