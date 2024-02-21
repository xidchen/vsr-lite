import cv2
import numpy as np

from backend import cfg


def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            x1 = xmin - cfg.SUBTITLE_AREA_DEVIATION_PIXEL
            if x1 < 0:
                x1 = 0
            y1 = ymin - cfg.SUBTITLE_AREA_DEVIATION_PIXEL
            if y1 < 0:
                y1 = 0
            x2 = xmax + cfg.SUBTITLE_AREA_DEVIATION_PIXEL
            y2 = ymax + cfg.SUBTITLE_AREA_DEVIATION_PIXEL
            cv2.rectangle(
                mask, (x1, y1), (x2, y2), (255, 255, 255),
                thickness=-1
            )
    return mask
