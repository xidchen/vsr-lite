import enum
import os
import logging
import warnings

import torch

warnings.filterwarnings("ignore")

logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STTN_MODEL_PATH = os.path.join(BASE_DIR, "models", "sttn", "infer_model.pth")

FFMPEG_PATH = "ffmpeg"


@enum.unique
class InpaintMode(enum.Enum):
    STTN = "sttn"
    LAMA = "lama"
    PROPAINTER = "propainter"


USE_H264 = True

MODE = InpaintMode.STTN
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
SUBTITLE_AREA_DEVIATION_PIXEL = 20
THRESHOLD_HEIGHT_DIFFERENCE = 20
PIXEL_TOLERANCE_X = 20
PIXEL_TOLERANCE_Y = 20

STTN_SKIP_DETECTION = True
STTN_NEIGHBOR_STRIDE = 5
STTN_REFERENCE_LENGTH = 10
STTN_MAX_LOAD_NUM = 50
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
