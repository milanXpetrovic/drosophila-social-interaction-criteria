import os

RAW_DATA = r"/srv/milky/drosophila-datasets/drosophila-isolation/data/trackings"
RAW_NORMALIZATION = r'/srv/milky/drosophila-datasets/drosophila-isolation/data/normalization'

TREATMENT = "CS_10D"

DATA_PREPARED = False

ARENA_DIEAMETER_MM = 61
FPS = 24 
START = 0
END = 20
EXP_DURATION = 20
TIMECUT = 0

ANGLE_BIN = 5
DISTANCE_BIN = 0.25
DISTANCE_MAX = 100

RANDOM_GROUP_SIZE = 15
N_RANDOM_1 = 500
N_RANDOM_2 = 500
MOVECUT = True
# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Input files
INPUT_DIR = os.path.join(ROOT_DIR, "data", "input", TREATMENT)
#TRACKINGS = os.path.join(INPUT_DIR, "trackings", TREATMENT)
TRACKINGS = os.path.join(INPUT_DIR, "trackings")
PXPERMM = os.path.join(INPUT_DIR, "pxpermm", f"{TREATMENT}.json")
NROMALIZATION = os.path.join(INPUT_DIR, "normalization.json")

OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "output")
