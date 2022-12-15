from pathlib import Path

TRAIN_DATA_PATH = Path('dataset/8_202_train.csv')
TEST_DATA_PATH = Path('dataset/8_202_test.csv')
FIG_FOLDER = Path('figures')
NUM_CLASSES = 2
LATENT_SIZE = 32
GEN_LEARNING_RATE = 0.01
DIS_LEARNING_RATE = 0.01
EPOCHS = 1000
