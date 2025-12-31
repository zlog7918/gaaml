POP_SIZE=100
CROSS_RATE=0.9
MUTATE_RATE=0.01
NUM_OF_GENERATIONS=500

# (name, (num_of_bits, min_v, max_v))
BIN_PART_LIST_LEN=('layer_num', (5, 1, 32))
# BIN_PART_LIST_LEN=('layer_num', (7, 1, 128))
BIN_PART_NEURON_NUM_SEED=('neuron_num_seed', (10, 0, 1023))
BIN_PART_NEURON_TYPE_SEED=('neuron_type_seed', (10, 0, 1023))
BIN_PART_REST=(
  ('x', (10, 1, 1000)),
  ('y', (10, 1, 1000)),
)

# (num_of_elem_bits, min_v, max_v)
NEURON_NUM=(20, 2, 1048577)
# NEURON_TYPE=(2, 0, 3)
NEURON_TYPE=(6, 0, 63)
