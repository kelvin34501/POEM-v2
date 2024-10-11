import os
import cv2

if __name__ == "__main__":
    # tune multi-threading params
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cv2.setNumThreads(0)

import random
import argparse
from argparse import Namespace
from time import time

import lib.models
from lib.models.model_abc import ModelABC
import numpy as np
import torch
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.config import CN
import json
from typing import Dict, Tuple, Optional

from server_tool.shared_memory import SharedMemoryManager
from server_tool.sync_unit import SyncUnit
import server_tool.socket_util as socket_util

import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from lib.utils.transform import _affine_transform, _affine_transform_post_rot

from transform.transform_jit import inv_transf, transf_point_array
from transform.transform_np import inv_transf_np, transf_point_array_np

def bbox_get_center_scale(bbox, expand=2.5, mindim=200):
    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    s = max(w, h)
    s = s * expand
    s = max(s, mindim)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return np.array((center_x, center_y)), s


def format_batch(img_list, bbox_list, req_flip, camera_name_list, cam_intr_map, cam_extr_map, output_size, device):
    cam_serial_list, cam_intr_list, cam_extr_list, image_list = [], [], [], []
    for cam_name, img, bbox in zip(camera_name_list, img_list, bbox_list):
        if bbox is None:
            continue
        cam_intr_ori = cam_intr_map[cam_name]
        cam_extr_ori = cam_extr_map[cam_name]
        # get bbox center and bbox scale
        bbox_center, bbox_scale = bbox_get_center_scale(bbox)
        affine = _affine_transform(center=bbox_center, scale=bbox_scale, out_res=output_size, rot=0)
        affine_2x3 = affine[:2, :]
        imgcrop = cv2.warpAffine(img,
                                 affine_2x3, (int(output_size[0]), int(output_size[1])),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        
        cv2.imshow(cam_name, imgcrop[..., ::-1])

        image = tvF.to_tensor(imgcrop)
        assert image.shape[0] == 3
        image = tvF.normalize(image, [0.5, 0.5, 0.5], [1, 1, 1])

        cc = np.array([cam_intr_ori[0, 2], cam_intr_ori[1, 2]])
        affine_postrot = _affine_transform_post_rot(center=bbox_center,
                                                    scale=bbox_scale,
                                                    optical_center=cc,
                                                    out_res=output_size,
                                                    rot=0)
        cam_intr = affine_postrot.dot(cam_intr_ori)

        cam_extr = cam_extr_ori  # TODO: req_flip

        image_list.append(image)
        cam_serial_list.append(cam_name)
        cam_intr_list.append(cam_intr)
        cam_extr_list.append(cam_extr)
    cv2.waitKey(1)
    if len(cam_serial_list) <= 1:
        return None

    cam_view_num = np.array(len(cam_serial_list))
    cam_intr_th = torch.as_tensor(np.stack(cam_intr_list, axis=0)).to(device)
    cam_extr_th = torch.as_tensor(np.stack(cam_extr_list, axis=0)).to(device)
    cam_transf_th = inv_transf(cam_extr_th)
    image_th = torch.stack(image_list, axis=0).to(device)
    master_id = torch.as_tensor(0).to(device)
    master_serial = cam_serial_list[0]

    # modified cam_transf_th --> master should be identity
    master_cam_transf_th = cam_transf_th[0].unsqueeze(0)
    target_cam_trasnf_th = inv_transf(master_cam_transf_th) @ cam_transf_th

    batch = {
        "image": image_th,  # (n, 3, RES_X, RES_Y)
        "cam_serial": [cam_serial_list],
        "cam_view_num": cam_view_num[None],  # (1, )
        "target_cam_intr": cam_intr_th[None],  # (1, ?, 3, 3)
        "target_cam_extr": target_cam_trasnf_th[None],  # (1, ?, 4, 4)
        "master_id": master_id[None],  # (1, )
        "master_serial": [master_serial],
    }
    return batch


def extract_pred(pred, batch, req_flip, cam_extr_map):
    # for k, v in pred.items():
    #     print(k, v.shape)
    master_id = batch["master_id"][0]
    master_serial = batch["master_serial"][0]
    joint3d_in_master = pred["pred_joints_3d"][0]
    joint3d_in_master_np = joint3d_in_master.detach().cpu().numpy()
    master_cam_extr = cam_extr_map[master_serial]
    master_cam_transf = inv_transf_np(master_cam_extr)
    joint3d_in_world_np = transf_point_array_np(master_cam_transf, joint3d_in_master_np)
    return {
        "joints": joint3d_in_world_np,
    }


def main(
    cfg: CN,
    arg: Namespace,
    time_f: float,
    camera_info: Dict[str, str],
    video_shape: Tuple[int, int],
    host: str,
    port: int,
):
    print("poem-v2 server start")
    # init socket
    s = socket_util.bind(host, port)

    # load model
    ## if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")
    device = torch.device(f"cuda:0")
    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=None, log_freq=arg.log_freq)
    model.to(device)
    model.eval()

    camera_name_list = list(camera_info.values())

    # load param
    cam_extr_map, cam_intr_map = {}, {}
    for cam_name in camera_name_list:
        _recv = SharedMemoryManager(
            name=f"cam_extr__{cam_name}",
            type=1,
            shape=(4, 4),
            dtype=np.float32,
            timeout=60,
        )
        _recv.unregister()
        cam_extr, _ = _recv.execute()
        _recv.close()
        cam_extr_map[cam_name] = cam_extr

        _recv = SharedMemoryManager(
            name=f"cam_intr__{cam_name}",
            type=1,
            shape=(3, 3),
            dtype=np.float32,
            timeout=60,
        )
        _recv.unregister()
        cam_intr, _ = _recv.execute()
        _recv.close()
        cam_intr_map[cam_name] = cam_intr

    # load (synced) image
    recv_list = []
    for cam_name in camera_name_list:
        _recv = SharedMemoryManager(
            name=f"sync__{cam_name}",
            type=1,
            shape=(video_shape[1], video_shape[0], 3),
            dtype=np.uint8,
            timeout=60,
        )
        _recv.unregister()
        recv_list.append(_recv)
    sync_unit = SyncUnit(recv_list)

    # start server
    while True:
        conn, addr = s.accept()
        with conn:
            inp = socket_util.conn_recv(conn)
            if inp is None:
                break
            # process
            img_list, ts = sync_unit.execute()
            payload = ts
            try:
                socket_util.conn_resp(conn, payload)
            except BrokenPipeError:
                pass

            inp2 = socket_util.conn_recv(conn)
            if inp2 is None:
                continue  # nothing detected

            start = time()
            ts, bboxes_left, bboxes_right = inp2

            # process_right
            batch = format_batch(img_list=img_list,
                                bbox_list=bboxes_right,
                                req_flip=False,
                                camera_name_list=camera_name_list,
                                cam_intr_map=cam_intr_map,
                                cam_extr_map=cam_extr_map,
                                output_size=cfg.DATA_PRESET.IMAGE_SIZE,
                                device=device)
            if batch is None:
                payload = None
            else:
                with torch.no_grad():
                    pred = model(batch, 0, "inference", epoch_idx=0)
                payload = extract_pred(pred, batch, req_flip=False, cam_extr_map=cam_extr_map)
            end = time()
            latency = int(end * 1000) - ts
            print("elapsed:", end - start, "latency", latency)

            try:
                socket_util.conn_resp(conn, payload)
            except BrokenPipeError:
                pass
    s.close()

    # close all
    for _recv in recv_list:
        _recv.close()

    print("poem-v2 server end")


MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server.camera_info", type=str, required=True)
    parser.add_argument("--server.video_shape", type=str, required=True)
    parser.add_argument("--server.host", type=str, default="localhost")
    parser.add_argument("--server.port", type=int, default=11312)
    # poem settings

    server_arg, _ = parser.parse_known_args()

    # parse camera_info
    camera_info_str = getattr(server_arg, "server.camera_info")
    camera_info = dict(el.split(":", 2) for el in camera_info_str.split(","))
    # parse video_shape
    video_shape_str = getattr(server_arg, "server.video_shape")
    _split = video_shape_str.split("x", 2)
    video_shape = (int(_split[0]), int(_split[1]))
    host = getattr(server_arg, "server.host")
    port = getattr(server_arg, "server.port")

    # poem
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # server
    main(cfg=cfg, arg=arg, time_f=exp_time, camera_info=camera_info, video_shape=video_shape, host=host, port=port)
