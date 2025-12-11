POP_SIZE=25
CROSS_RATE=0.9
MUTATE_RATE=0.01
NUM_OF_GENERATIONS=100

# (name, (num_of_bits, min_v, max_v))
BIN_PART_LIST_LEN=('layer_num', (7, 1, 128))
BIN_PART_REST=(
  ('neuron_num_seed', (10, 0, 1023)),
  ('neuron_type_seed', (10, 0, 1023)),
  ('optimizer', (1, 0, 1)),
  ('learning_rate', (13, 1, 8192)),
  # TODO: maybe this version [2^(-13) - 1.] → [2^(-13) - 0.5]
  # ('learning_rate', (12, 1, 4096)),
  ('epochs', (10, 1, 1024)),
)

# (num_of_elem_bits, min_v, max_v)
NEURON_NUM=(20, 2, 1048577)
NEURON_TYPE=(2, 0, 3)