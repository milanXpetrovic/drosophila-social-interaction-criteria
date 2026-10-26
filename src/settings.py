from pathlib import Path

from settings import main_settings

# RAW_DATA = r"/srv/milky/drosophila-datasets/drosophila-isolation/data/trackings"
# RAW_NORMALIZATION = r'/srv/milky/drosophila-datasets/drosophila-isolation/data/normalization'

TREATMENT = "CS_old3-5d_24group"

ROOT_DIR = main_settings.SAVE_DATA_LOCATION
# ROOT_DIR = Path('/srv/milky/drosophila-pipeline')



INPUT_DIR = ROOT_DIR / "preproc-trajectories/normalize_trajectories" / TREATMENT


INTERACTION_CRITERIA = ROOT_DIR / '4_drosophila_social_interaction_criteria'

# TODO: fix logs location to be with script
LOGS_DIR = INTERACTION_CRITERIA / "logs"
OUTPUT_DIR = INTERACTION_CRITERIA / "data"  # / "output"
TRACKINGS = INTERACTION_CRITERIA / "trackings" / TREATMENT
PXPERMM = INTERACTION_CRITERIA / "pxpermm" / f"{TREATMENT}.json"

NROMALIZATION = INTERACTION_CRITERIA / "data" / TREATMENT / "normalization.json"


DATA_PREPARED = False
MOVECUT = True

ARENA_DIAMETER_MM = 81
FPS = 24
START = 0
END = 20
EXP_DURATION = 20
TIMECUT = 0

ANGLE_BIN = 5
DISTANCE_BIN = 0.25
DISTANCE_MAX = 100

RANDOM_GROUP_SIZE = 24
N_RANDOM_1 = 500
N_RANDOM_2 = 500
