POP_SIZE=100
CROSS_RATE=0.9
MUTATE_RATE=0.01
NUM_OF_GENERATIONS=500

BIN_PART_OPTIMIZER_NAME='optimizer'
BIN_PART_LEARNING_RATE_NAME='learning_rate'
BIN_PART_BATCH_NAME='batch_size'
BIN_PART_EPOCHS_NAME='epochs'

# (name, (num_of_bits, min_v, max_v))
BIN_PART_LIST_LEN=('layer_num', (5, 1, 32))
BIN_PART_NEURON_NUM_SEED=('neuron_num_seed', (10, 0, 1023))
BIN_PART_NEURON_TYPE_SEED=('neuron_type_seed', (10, 0, 1023))
BIN_PART_REST=(
  (BIN_PART_OPTIMIZER_NAME, (1, 0, 1)),
  (BIN_PART_LEARNING_RATE_NAME, (12, 1, 4096)),
  # TODO: maybe this version [2^(-13) - 0.5] → [2^(-13) - 1.]
  # (BIN_PART_LEARNING_RATE_NAME, (13, 1, 8192)),
  (BIN_PART_EPOCHS_NAME, (7, 1, 128)),
  (BIN_PART_BATCH_NAME, (10, 1, 1024)),
)

# (num_of_elem_bits, min_v, max_v)
NEURON_NUM=(10, 2, 724)
NEURON_TYPE=(2, 0, 3)
