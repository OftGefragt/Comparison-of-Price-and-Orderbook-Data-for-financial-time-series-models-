EMBEDDING_DIM = 256
TOP_K = 3
TIME_STEP = 10
PERCENTILE = 95

TRAIN_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LEARNING_RATE_CENTER = 0.001
N_LAYERS = 2
DROPOUT_RATE = 0.2
DEVICE = 'mps'

TRAIN_PATH = '../data/train/BTC_1min.csv'
TEST_PATH = '../data/test/ETH_1min.csv'
SVDD_PATH = '../model/svdd_center.pth'
GRU_ENCODER_PATH = '../model/gru_encoder.pth'