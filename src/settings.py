from pathlib import Path

RAW_DATA = r"/srv/milky/drosophila-datasets/drosophila-isolation/data/trackings"
RAW_NORMALIZATION = r'/srv/milky/drosophila-datasets/drosophila-isolation/data/normalization'

ROOT_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT_DIR / "logs"
INPUT_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "data" / "output"

TREATMENT = "CsCh"
DATA_PREPARED = False
MOVECUT = True

ARENA_DIAMETER_MM = 61
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

TRACKINGS = INPUT_DIR / "trackings" / TREATMENT
PXPERMM = INPUT_DIR / "pxpermm" / f"{TREATMENT}.json"
NORMALIZATION = INPUT_DIR / "normalization.json"
