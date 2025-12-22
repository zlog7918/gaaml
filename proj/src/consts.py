POP_SIZE=25
CROSS_RATE=0.9
MUTATE_RATE=0.01
NUM_OF_GENERATIONS=100

# (name, (num_of_bits, min_v, max_v))
BIN_PART_LIST_LEN=('layer_num', (7, 1, 128))
BIN_PART_REST=(
  ('x', (10, 1, 1000)),
  ('y', (10, 1, 1000)),
)

# (num_of_elem_bits, min_v, max_v)
NEURON_NUM=(20, 2, 1048577)

MIN_MAX_LEN=(2, 9)
NEURON_TYPE=(6, 0, 63)
# NEURON_TYPE=(2, 0, 3)