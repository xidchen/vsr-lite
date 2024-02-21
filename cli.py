import argparse
import importlib
import multiprocessing
import os
import pathlib
import platform
import sys
import tempfile
import threading
import time
import shutil
import subprocess

import cv2
import torch
import tqdm

from backend import cfg
from backend.inpaint.sttn_inpaint import STTNVideoInpaint
from backend.tools.inpaint_tools import create_mask


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubtitleRemover:

    def __init__(self, vd_path, sub_area=None, gui_mode=False):
        importlib.reload(cfg)
        self.lock = threading.RLock()
        self.sub_area = sub_area
        self.gui_mode = gui_mode
        self.is_picture = False
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        self.vd_name = pathlib.Path(self.video_path).stem
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.size = (self.frame_width, self.frame_height)
        self.mask_size = (self.frame_height, self.frame_width)
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self.video_writer = cv2.VideoWriter(
            self.video_temp_file.name, cv2.VideoWriter.fourcc(*"mp4v"),
            self.fps, self.size
        )
        self.video_out_name = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self.video_path))),
            "videos_no_sub",
            f"{self.vd_name}_no_sub.mp4"
        )
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if torch.cuda.is_available():
            print("Use GPU for acceleration")
        self.progress_total = 0
        self.progress_remover = 0
        self.is_finished = False
        self.preview_frame = None
        self.is_successful_merged = False

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def sttn_mode_with_no_detection(self, tbar):
        print("Use sttn mode with no detection")
        print("[Processing] Start removing subtitles...")
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print(
                "[Info] No subtitle area has been set. "
                "Video will be processed in full screen. "
                "As a result, the final outcome might be suboptimal."
            )
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def sttn_mode(self, tbar):
        if cfg.STTN_SKIP_DETECTION:
            self.sttn_mode_with_no_detection(tbar)

    def run(self):
        start_time = time.time()
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.progress_total = 0
        tbar = tqdm.tqdm(
            total=int(self.frame_count),
            unit="frame",
            position=0,
            file=sys.__stdout__,
            desc="Subtitle Removing"
        )
        print()
        if cfg.MODE == cfg.InpaintMode.STTN:
            self.sttn_mode(tbar)
        self.video_cap.release()
        self.video_writer.release()
        print()
        if not self.is_picture:
            self.merge_audio_to_video()
            print("[Finished] Subtitle removed.")
            print(f"[Finished] Video generated at: {self.video_out_name}")
        else:
            print("[Finished] Subtitle removed.")
            print(f"[Finished] Picture generated at: {self.video_out_name}")
        time_cost = time.time() - start_time
        hours, _ = divmod(time_cost, 3600)
        minutes, seconds = divmod(time_cost, 60)
        if hours:
            print(f"Time cost: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        elif minutes:
            print(f"Time cost: {int(minutes)}m {int(seconds)}s")
        elif seconds:
            print(f"Time cost: {seconds}s")
        self.is_finished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except OSError:
                if platform.system() in ["Windows"]:
                    pass
                else:
                    print(f"Failed to delete temp file {self.video_temp_file.name}")

    def merge_audio_to_video(self):
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [
            cfg.FFMPEG_PATH,
            "-y", "-i", self.video_path,
            "-acodec", "copy", "-vn", "-loglevel", "error", temp.name
        ]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(
                audio_extract_command, stdin=open(os.devnull), shell=use_shell
            )
        except subprocess.CalledProcessError:
            print("Fail to extract audio")
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [
                    cfg.FFMPEG_PATH,
                    "-y", "-i", self.video_temp_file.name, "-i", temp.name,
                    "-vcodec", "libx264" if cfg.USE_H264 else "copy",
                    "-acodec", "copy", "-loglevel", "error", self.video_out_name
                ]
                try:
                    subprocess.check_output(
                        audio_merge_command, stdin=open(os.devnull), shell=use_shell
                    )
                except subprocess.CalledProcessError:
                    print("Fail to merge audio")
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except OSError:
                    if platform.system() in ["Windows"]:
                        pass
                    else:
                        print(f"Failed to delete temp file {temp.name}")
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print(f"Unable to copy file. {e}")
            self.video_temp_file.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(description="video subscript remover")
    parser.add_argument(
        "--file",
        type=str, help="video absolute path"
    )
    parser.add_argument(
        "--area",
        default=(1450, 1600, 180, 900),
        type=tuple[int], help="vubtitle area"
    )
    args = vars(parser.parse_args())
    video_file_path = args["file"]
    if not os.path.exists(video_file_path):
        print(f"File does not exist: {video_file_path}")
    subtitle_area = args["area"]
    if len(subtitle_area) != 4:
        print(f"Subtitle area not correct: {subtitle_area}")
    sd = SubtitleRemover(video_file_path, sub_area=subtitle_area)
    sd.run()
