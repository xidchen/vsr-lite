import copy
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision

from backend import cfg
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor


_to_tensors = torchvision.transforms.Compose([
    Stack(),
    ToTorchFormatTensor()
])


class STTNInpaint:

    def __init__(self):
        self.device = cfg.device
        self.model = InpaintGenerator().to(self.device)
        self.model.load_state_dict(
            torch.load(cfg.STTN_MODEL_PATH, map_location=self.device)["netG"]
        )
        self.model.eval()
        self.model_input_width, self.model_input_height = 640, 120
        self.neighbor_stride = cfg.STTN_NEIGHBOR_STRIDE
        self.ref_length = cfg.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        h_ori, w_ori = mask.shape[:2]
        h_ori = int(h_ori + 0.5)
        w_ori = int(w_ori + 0.5)
        split_h = int(w_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(h_ori, split_h, mask)
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}
        comps = {}
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(
                    image_crop,
                    dsize=(self.model_input_width, self.model_input_height)
                )
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.inpaint(frames_scaled[k])
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], dsize=(w_ori, split_h))
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                        mask_area * comp +
                        (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    )
                inpainted_frames.append(frame)
                print(f"processing frame, {len(frames_hr) - j} left")
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        frame_length = len(frames)
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        feats = feats.to(self.device)
        comp_frames = [None] * frame_length
        with torch.no_grad():
            feats = self.model.encoder(
                feats.view(frame_length, 3, self.model_input_height, self.model_input_width)
            )
            _, c, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        for f in range(0, frame_length, self.neighbor_stride):
            neighbor_ids = [
                i for i in range(
                    max(0, f - self.neighbor_stride),
                    min(frame_length, f + self.neighbor_stride + 1)
                )
            ]
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            with (torch.no_grad()):
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(
                    self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])
                ).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            np.array(comp_frames[idx]).astype(np.float32) * 0.5 +
                            img.astype(np.float32) * 0.5
                        )
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(h_ori, split_h, mask):
        inpaint_area = []
        to_h = from_h = h_ori
        while from_h != 0:
            if to_h - split_h < 0:
                from_h = 0
                to_h = split_h
            else:
                from_h = to_h - split_h
            if not np.all(mask[from_h:to_h, :] == 0) and np.sum(mask[from_h:to_h, :]) > 10:
                if to_h != h_ori:
                    move = 0
                    while to_h + move < h_ori and not np.all(mask[to_h + move, :] == 0):
                        move += 1
                    if to_h + move < h_ori and move < split_h:
                        to_h += move
                        from_h += move
                if (from_h, to_h) not in inpaint_area:
                    inpaint_area.append((from_h, to_h))
                else:
                    break
            to_h -= split_h
        return inpaint_area


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        reader = cv2.VideoCapture(self.video_path)
        frame_info = {
            "W_ori": int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            "H_ori": int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            "fps": reader.get(cv2.CAP_PROP_FPS),
            "len": int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        }
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None):
        self.sttn_inpaint = STTNInpaint()
        self.video_path = video_path
        self.mask_path = mask_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(self.video_path))),
            "videos_no_sub",
            f"{os.path.basename(self.video_path).rsplit('.')[0]}_no_sub.mp4"
        )
        if clip_gap is None:
            self.clip_gap = cfg.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(
            self, input_mask=None, input_sub_remover=None,
            tbar=None, start_frame=0, end_frame=None
    ):
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            writer = cv2.VideoWriter(
                self.video_out_path,
                cv2.VideoWriter.fourcc(*"mp4v"),
                frame_info["fps"],
                (frame_info["W_ori"], frame_info["H_ori"])
            )
        rec_time = (
            frame_info["len"] // self.clip_gap
            if frame_info["len"] % self.clip_gap == 0
            else frame_info["len"] // self.clip_gap + 1
        )
        split_h = int(frame_info["W_ori"] * 3 / 16)
        if input_mask is None:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(
            frame_info["H_ori"], split_h, mask
        )
        for i in range(rec_time):
            for j in range(self.clip_gap):
                success, image = reader.read()
                if not success:
                    break
                current_frame = i * self.clip_gap + j
                original_frame = image.copy()
                if start_frame <= current_frame < end_frame:
                    frames = [cv2.resize(image[inpaint_area[k][0]:inpaint_area[k][1], :, :],
                                         dsize=(self.sttn_inpaint.model_input_width,
                                                self.sttn_inpaint.model_input_height))
                              for k in range(len(inpaint_area))]
                    comps = {
                        k: self.sttn_inpaint.inpaint([frames[k]])[0]
                        for k in range(len(inpaint_area))
                    }
                    for k in range(len(inpaint_area)):
                        comp = cv2.resize(comps[k], dsize=(frame_info["W_ori"], split_h))
                        comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                        image[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                            mask_area * comp +
                            (1 - mask_area) * image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        )
                writer.write(image)
                if input_sub_remover is not None:
                    if tbar is not None:
                        input_sub_remover.update_progress(tbar, increment=1)
                    if input_sub_remover.gui_mode:
                        input_sub_remover.preview_frame = cv2.hconcat([original_frame, image])
        writer.release()
