import pytest
from gaaml.classes import _utils as util
from gaaml.classes.NetIndividual import NetIndividual as NI
from gaaml.classes.MaxIntsIndividual import MaxIntsIndividual as _MII
from gaaml.classes.MaxIntsListIndividual import MaxIntsListIndividual as _MILI

def test_type_len_modifier() -> None:
  # values
  for l, exp_l in (
    (4, 6),
    (7, 9),
    (16, 18),
    (1, 3),
  ):
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
  len_name, (len_bit, len_min, len_max)=layers_len
  bit_len1=ni1.gen[0].fenotype[len_name]
  bit_len2=ni2.gen[0].fenotype[len_name]

  num_seed_name, (num_seed_bit, num_seed_min, num_seed_max)=num_seed
  type_seed_name, (type_seed_bit, type_seed_min, type_seed_max)=type_seed
  (
    (x_name, (x_bit, x_min, x_max)),
    (y_name, (y_bit, y_min, y_max)),
  )=g_schema
  n_bit, n_min, n_max=n_schema
  t_bit, t_min, t_max=t_schema
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

  bits_t1: 1 0  1 1  0 1
                      ↑
  bits_t2: 0 1  1 0  0 1  0 0
                 ↑
  cp1:5,3

  expected:
  bits0: 0 1  0 1  0 0  0 0  0 1 1
  bits_n: 1 1 1
  bits_t: 1 0  1 1  0 0  0 1  0 0
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
  ni1.gen[0].gen._gen=bytearray('0 0  1 0  1 1  1 0  1 0 1'.replace(' ', '').encode())
  ni1.gen[1].gen._gen=bytearray('1 1 0'.replace(' ', '').encode())
  ni1.gen[2].gen._gen=bytearray('1 0  1 1  0 1'.replace(' ', '').encode())
  ni2=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni2.gen[0].gen._gen=bytearray('0 1  0 1  0 0  0 0  0 1 1'.replace(' ', '').encode())
  ni2.gen[1].gen._gen=bytearray('0 1 0  0 0 1'.replace(' ', '').encode())
  ni2.gen[2].gen._gen=bytearray('0 1  1 0  0 1  0 0'.replace(' ', '').encode())
  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema

  for cp, (exp_str0, exp_str1, exp_str2), (exp_feno0, exp_feno1, exp_feno2) in (
    (
      (1, (2, 5), (5,3)), (
        '0 1  0 1  0 0  0 0  0 1 1',
        '1 1 1',
        '1 0  1 1  0 0  0 1  0 0',
      ), (
        {len_name: 2, num_seed_name: 1, type_seed_name: 0, x_name: 1, y_name: 3},
        [7, 3], # one more: len->2, filled in by random with num_seed(1)->3
        [2, 3, 0, 1], # one less: len->2 -> 2+2=4
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

  bits_t1: 1 0  1 1  0 1
            ↑
  bits_t2: 0 1  1 0  0 1  0 0
                 ↑
  cp1:1,3

  expected:
  bits0: 0 1  0 1  0 0  0 0  0 1 1
  bits_n: 1 1 1
  bits_t: 1 0  1 1  0 0  0 1  0 0
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
  ni1.gen[0].gen._gen=bytearray('0 0  1 0  1 1  1 0  1 0 1'.replace(' ', '').encode())
  ni1.gen[1].gen._gen=bytearray('1 1 0'.replace(' ', '').encode())
  ni1.gen[2].gen._gen=bytearray('1 0  1 1  0 1'.replace(' ', '').encode())
  ni2=NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni2.gen[0].gen._gen=bytearray('0 1  0 1  0 0  0 0  0 1 1'.replace(' ', '').encode())
  ni2.gen[1].gen._gen=bytearray('0 1 0  0 0 1'.replace(' ', '').encode())
  ni2.gen[2].gen._gen=bytearray('0 1  1 0  0 1  0 0'.replace(' ', '').encode())
  len_name, (len_bit, len_min, len_max)=layers_len

  num_seed_name, (num_seed_bit, num_seed_min, num_seed_max)=num_seed
  type_seed_name, (type_seed_bit, type_seed_min, type_seed_max)=type_seed
  (
    (x_name, (x_bit, x_min, x_max)),
    (y_name, (y_bit, y_min, y_max)),
  )=g_schema
  n_bit, n_min, n_max=n_schema
  t_bit, t_min, t_max=t_schema
  gen0_bit=len_bit+num_seed_bit+type_seed_bit+x_bit+y_bit

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
          '1 0  0 1  0 0',
        ),
        (
          {len_name: 2, num_seed_name: 1, type_seed_name: 0, x_name: 1, y_name: 3},
          [7, 3], # one more: len->2, filled in by random with num_seed(1)->3
          [2, 1, 0, 3], # one more: len->2, filled in by random with type_seed(0)->3
        ),
      ), (
        (
          '0 0  1 0  1 1  1 0  1 0 1',
          '0 1 0  0 0 0',
          '0 1  1 0  1 1  0 1',
        ),
        (
          {len_name: 1, num_seed_name: 2, type_seed_name: 3, x_name: 3, y_name: 5},
          [4], # one less: len->1
          [1, 2, 3], # one less: len->1 -> 1+2=3
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
  bits_t: 0 1  1 0  0 1  0 0

  expected:
  fenotype0: {
    len: 1+1=2
    n_s: 2+0
    t_s: 2+0
    x: 2+1=3
    y: 6+0=6 -> 6->4 -> 4+0=4
  }
  fenotype1: [2+2->4, 1+2->3]
  fenotype2: [1+0->1, 2+0->2, 1+0->1, 0+0->0]
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
        ([2, 3, 0, 1], True), # one less: len->2 -> 2+2=4
      )
    ),
    (
      (
        '1 1  0 1  0 1  0 1  0 1 1',
        '1 1 1  1 0 1  0 1 1  0 0 0',
        '1 0  1 1  0 0',
      ), (
        {len_name: 4, num_seed_name: 1, type_seed_name: 1, x_name: 2, y_name: 3},
        ([7, 7, 5, 2], False),
        ([2, 3, 0, 1, 0, 2], True), # three more: len->4 -> 4+2=6, filled in by random with num_seed(1)->[1, 0, 2]
      )
    ),
  ):
    for i, _str in zip(range(3), (str0, str1, str2)):
      ni.gen[i].gen._gen=bytearray(_str.replace(' ', '').encode())
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

def test_error_not_same_conf_on_get_cp() -> None:
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
  ni1=NI(*schema1)

  for schema2 in (
    (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI.get_cp(ni1, ni2)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_gen_size_on_get_cp() -> None:
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
  ni1=NI(*schema1)

  for schema2 in (
    (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI.get_cp(ni1, ni2)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_conf_on_create() -> None:
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
  ni1=NI(*schema1)
  dummy_cp=1, (1, 1), (1, 1)

  for schema2 in (
    (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI(ni1, ni2, cross_point=dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_gen_size_on_create() -> None:
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
  ni1=NI(*schema1)
  dummy_cp=1, (1, 1), (1, 1)

  for schema2 in (
    (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI(ni1, ni2, cross_point=dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_conf_on_crossover() -> None:
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
  ni1=NI(*schema1)
  dummy_cp=1, (1, 1), (1, 1)

  for schema2 in (
    (('len', (2, 2, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (2, 1, 3)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len', (3, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (('len2', (2, 1, 4)), num_seed, type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 0, 2)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (2, 1, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed', (3, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, ('num_seed2', (2, 0, 3)), type_seed, (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 0, 2)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (2, 1, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed', (3, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, ('type_seed2', (2, 0, 3)), (g_schema, n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('y', (3, 0, 5)),), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 3)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 2, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (3, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x2', (2, 1, 4)),('y', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y2', (3, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (4, 0, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 1, 5))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, ((('x', (2, 1, 4)),('y', (3, 0, 5)),('z', (3, 0, 6))), n_schema, t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 2, 8), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, (3, 1, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 0, 2))),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (2, 1, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI.crossover(ni1, ni2, dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_gen_size_on_crossover() -> None:
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
  ni1=NI(*schema1)
  dummy_cp=1, (1, 1), (1, 1)

  for schema2 in (
    (layers_len, num_seed, type_seed, (g_schema, (4, 2, 7), t_schema)),
    (layers_len, num_seed, type_seed, (g_schema, n_schema, (3, 0, 3))),
    # (layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema)),
  ):
    ni2=NI(*schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI.crossover(ni1, ni2, dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_outside_range_on_create() -> None:
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

  for cp in (
    (-1, (1, 1), (1, 1)),
    (gen0_bit+1, (1, 1), (1, 1)),
    (1, (-1, -1%n_bit), (1, 1)),
    (1, (-1%n_bit, -1), (1, 1)),
    (1, (1, 1), (-1, -1%t_bit)),
    (1, (1, 1), (-1%t_bit, -1)),
    (1, (len1*n_bit+1, 1), (1, 1)),
    (1, (1, len2*n_bit+1), (1, 1)),
    (1, (1, 1), (NI.type_len_modifier(len1)*t_bit+1, 1)),
    (1, (1, 1), (1, NI.type_len_modifier(len2)*t_bit+1)),
    # (1, (1, 1), (1, 1)),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI(ni1, ni2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_outside_range_on_crossover() -> None:
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

  for cp in (
    (-1, (1, 1), (1, 1)),
    (gen0_bit+1, (1, 1), (1, 1)),
    (1, (-1, -1%n_bit), (1, 1)),
    (1, (-1%n_bit, -1), (1, 1)),
    (1, (1, 1), (-1, -1%t_bit)),
    (1, (1, 1), (-1%t_bit, -1)),
    (1, (len1*n_bit+1, 1), (1, 1)),
    (1, (1, len2*n_bit+1), (1, 1)),
    (1, (1, 1), (NI.type_len_modifier(len1)*t_bit+1, 1)),
    (1, (1, 1), (1, NI.type_len_modifier(len2)*t_bit+1)),
    # (1, (1, 1), (1, 1)),
  ):
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

def test_error_too_short_on_create_from_two() -> None:
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

  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema
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
      ni.gen[i].gen._gen=bytearray(_str.replace(' ', '').encode())
      ni.gen[i]._update_fenotype()
    ni._update()

  for cp in (
    (1, (0, 3), (1, 1)),
    (1, (1, 4), (1, 1)),
    (1, (2, 5), (1, 1)),
    (1, (3, 6), (1, 1)),
    (1, (1, 1), (0, 2)),
    (1, (1, 1), (1, 3)),
    (1, (1, 1), (4, 6)),
    (1, (1, 1), (4, 8)),
    (1, (1, 1), (3, 7)),
    (1, (1, 1), (1, 7)),
    (1, (1, 1), (1, 5)),
    # (1, (1, 1), (1, 1)),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI(ni1, ni2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_too_short_on_crossover() -> None:
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

  len_name, _=layers_len
  num_seed_name, _=num_seed
  type_seed_name, _=type_seed
  (
    (x_name, _),
    (y_name, _),
  )=g_schema
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
      ni.gen[i].gen._gen=bytearray(_str.replace(' ', '').encode())
      ni.gen[i]._update_fenotype()
    ni._update()

  for cp in (
    # first too short
    (1, (0, 3), (1, 1)),
    (1, (1, 4), (1, 1)),
    (1, (2, 5), (1, 1)),
    (1, (3, 6), (1, 1)),
    (1, (1, 1), (0, 2)),
    (1, (1, 1), (1, 3)),
    (1, (1, 1), (4, 6)),
    (1, (1, 1), (4, 8)),
    (1, (1, 1), (3, 7)),
    (1, (1, 1), (1, 7)),
    (1, (1, 1), (1, 5)),
    # second too short
    (1, (3, 0), (1, 1)),
    (1, (4, 1), (1, 1)),
    (1, (5, 2), (1, 1)),
    (1, (6, 3), (1, 1)),
    (1, (1, 1), (2, 0)),
    (1, (1, 1), (3, 1)),
    (1, (1, 1), (6, 4)),
    (1, (1, 1), (8, 4)),
    (1, (1, 1), (7, 3)),
    (1, (1, 1), (7, 1)),
    (1, (1, 1), (5, 1)),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI.crossover(ni1, ni2, cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_illegal_argument_on_create() -> None:
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

  for args, kwargs in (
    ((layers_len, num_seed), {}),
    ((layers_len, num_seed, None), {}),
    ((layers_len, num_seed, None, None), {}),
    ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {}),
    ((layers_len, num_seed, type_seed), {}),
    ((layers_len, num_seed, type_seed, None), {}),
    ((layers_len, ni2), {}),
    ((layers_len, ni2, None), {}),
    ((layers_len, ni2, None, None), {}),
    ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {}),
    ((layers_len, ni2, type_seed), {}),
    ((layers_len, ni2, type_seed, None), {}),
    ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {}),
    ((ni1, num_seed), {}),
    ((ni1, num_seed, None), {}),
    ((ni1, num_seed, None, None), {}),
    ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {}),
    ((ni1, num_seed, type_seed), {}),
    ((ni1, num_seed, type_seed, None), {}),
    ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {}),
    ((layers_len, num_seed), {'cross_point': dummy_cp}),
    ((layers_len, num_seed, None), {'cross_point': dummy_cp}),
    ((layers_len, num_seed, None, None), {'cross_point': dummy_cp}),
    ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    ((layers_len, num_seed, type_seed), {'cross_point': dummy_cp}),
    ((layers_len, num_seed, type_seed, None), {'cross_point': dummy_cp}),
    ((layers_len, ni2), {'cross_point': dummy_cp}),
    ((layers_len, ni2, None), {'cross_point': dummy_cp}),
    ((layers_len, ni2, None, None), {'cross_point': dummy_cp}),
    ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    ((layers_len, ni2, type_seed), {'cross_point': dummy_cp}),
    ((layers_len, ni2, type_seed, None), {'cross_point': dummy_cp}),
    ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    ((ni1, num_seed), {'cross_point': dummy_cp}),
    ((ni1, num_seed, None), {'cross_point': dummy_cp}),
    ((ni1, num_seed, None, None), {'cross_point': dummy_cp}),
    ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    ((ni1, num_seed, type_seed), {'cross_point': dummy_cp}),
    ((ni1, num_seed, type_seed, None), {'cross_point': dummy_cp}),
    ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': dummy_cp}),
    ((layers_len, num_seed), {'cross_point': None}),
    ((layers_len, num_seed, None), {'cross_point': None}),
    ((layers_len, num_seed, None, None), {'cross_point': None}),
    ((layers_len, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    ((layers_len, num_seed, type_seed), {'cross_point': None}),
    ((layers_len, num_seed, type_seed, None), {'cross_point': None}),
    ((layers_len, ni2), {'cross_point': None}),
    ((layers_len, ni2, None), {'cross_point': None}),
    ((layers_len, ni2, None, None), {'cross_point': None}),
    ((layers_len, ni2, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    ((layers_len, ni2, type_seed), {'cross_point': None}),
    ((layers_len, ni2, type_seed, None), {'cross_point': None}),
    ((layers_len, ni2, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    ((ni1, num_seed), {'cross_point': None}),
    ((ni1, num_seed, None), {'cross_point': None}),
    ((ni1, num_seed, None, None), {'cross_point': None}),
    ((ni1, num_seed, None, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    ((ni1, num_seed, type_seed), {'cross_point': None}),
    ((ni1, num_seed, type_seed, None), {'cross_point': None}),
    ((ni1, num_seed, type_seed, (g_schema, n_schema, t_schema)), {'cross_point': None}),
    ((ni1, ni2), {}),
    ((ni1, ni2), {'cross_point': None}),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=NI(*args, **kwargs) # type: ignore

    # results
    assert str(excinfo.value)=='Illegal argument options'
