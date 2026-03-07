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
from time import time, sleep

import lib.models
from lib.models.model_abc import ModelABC
import numpy as np

# NumPy compatibility fix for numpy._core module
import sys
try:
    import numpy._core
except ImportError:
    # Create a compatibility module for older NumPy versions
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    sys.modules['numpy._core._multiarray_umath'] = _core._multiarray_umath

import torch
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.config import CN
import msgpack
import msgpack_numpy
import yaml
import pickle
import base64
import json
from typing import Dict, Tuple, Optional, List
import zmq
import atexit
import threading
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from lib.utils.transform import _affine_transform, _affine_transform_post_rot

from transform.transform_jit import inv_transf, transf_point_array
from transform.transform_np import inv_transf_np, transf_point_array_np, project_point_array_np

from tool.flip_util import flip_cam_extr
from server_tool.timestamp_buffer import TimestampBuffer


@dataclass
class SyncFrameCache:
    """Cache for synchronized frame data."""
    timestamp: float
    img_list: List[np.ndarray]
    processed: bool = False


class SyncReceiverThread(threading.Thread):
    """Thread for receiving synchronized images from sync_channel."""

    def __init__(
        self,
        sync_socket: zmq.Socket,
        frame_buffer: TimestampBuffer,
        video_shape: Tuple[int, int],
    ):
        super().__init__(daemon=True)
        self._sync_socket = sync_socket
        self._frame_buffer = frame_buffer
        self._video_shape = video_shape
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        """Main loop for receiving synchronized images."""
        poller = zmq.Poller()
        poller.register(self._sync_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                # Poll with timeout to allow checking stop event
                socks = dict(poller.poll(timeout=10))  # 10ms timeout
                if self._sync_socket not in socks:
                    continue

                sync_msg = self._sync_socket.recv(zmq.NOBLOCK)
                sync_data = msgpack.unpackb(sync_msg, object_hook=msgpack_numpy.decode, raw=False)
                synced_data = sync_data.get("synced_data")
                synced_ts = sync_data.get("synced_ts")

                if synced_data is not None and synced_ts is not None:
                    # Extract color images from synced_data
                    img_list = []
                    for payload in synced_data:
                        if isinstance(payload, dict) and "color" in payload:
                            color_img = payload["color"].reshape(self._video_shape[1], self._video_shape[0], 3)
                            img_list.append(color_img)
                    if img_list:
                        self._frame_buffer.add(synced_ts,
                                               SyncFrameCache(
                                                   timestamp=synced_ts,
                                                   img_list=img_list,
                                                   processed=False,
                                               ))
            except zmq.Again:
                pass
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"SyncReceiverThread error: {e}")


def bbox_get_center_scale(bbox, expand=1.4, mindim=150):
    w, h = float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
    s = max(w, h)
    s = s * expand
    s = max(s, mindim)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return np.array((center_x, center_y)), s


def format_batch(img_list, bbox_list, req_flip, camera_name_list, cam_intr_map, cam_extr_map, img_size, output_size,
                 device):
    cam_serial_list, cam_intr_list, cam_extr_list, image_list = [], [], [], []
    for cam_name, img, bbox in zip(camera_name_list, img_list, bbox_list):
        if bbox is None:
            continue
        cam_intr_ori = cam_intr_map[cam_name]
        cam_extr_ori = cam_extr_map[cam_name]
        # get bbox center and bbox scale
        bbox_center, bbox_scale = bbox_get_center_scale(bbox)
        cam_center = np.array([cam_intr_ori[0, 2], cam_intr_ori[1, 2]])

        if req_flip:
            bbox_center[0] = 2 * cam_center[0] - bbox_center[0]
            # image & mask should be flipped horizontally with center at cam_center[0]
            # use cv2
            M = np.array([[-1, 0, 2 * cam_center[0]], [0, 1, 0]], dtype=np.float32)
            # Use warpAffine to apply the reflection
            img = cv2.warpAffine(img, M, img_size)
            cam_extr = inv_transf_np(flip_cam_extr(inv_transf_np(cam_extr_ori))).astype(np.float32)
        else:
            cam_extr = cam_extr_ori.copy()

        affine = _affine_transform(center=bbox_center, scale=bbox_scale, out_res=output_size, rot=0)
        affine_2x3 = affine[:2, :]
        imgcrop = cv2.warpAffine(img,
                                 affine_2x3, (int(output_size[0]), int(output_size[1])),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)

        # cv2.imshow(cam_name + '-' + str(req_flip), imgcrop[..., ::-1])

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

        image_list.append(image)
        cam_serial_list.append(cam_name)
        cam_intr_list.append(cam_intr)
        cam_extr_list.append(cam_extr)
    # cv2.waitKey(1)
    if len(cam_serial_list) <= 1:
        return None

    cam_view_num = np.array(len(cam_serial_list))
    cam_intr_th = torch.as_tensor(np.stack(cam_intr_list, axis=0)).to(device)
    cam_extr_th = torch.as_tensor(np.stack(cam_extr_list, axis=0)).to(device)
    cam_transf_th = inv_transf(cam_extr_th)
    image_th = torch.stack(image_list, dim=0).to(device)
    master_id = torch.as_tensor(0).to(device)
    master_serial = cam_serial_list[0]

    # modified cam_transf_th --> master should be identity
    master_cam_transf_th = cam_transf_th[0].unsqueeze(0)
    target_cam_trasnf_th = inv_transf(master_cam_transf_th) @ cam_transf_th

    batch = {
        "image": image_th.float(),  # (n, 3, RES_X, RES_Y)
        "cam_serial": [cam_serial_list],
        "cam_view_num": cam_view_num[None],  # (1, )
        "target_cam_intr": cam_intr_th[None].float(),  # (1, ?, 3, 3)
        "target_cam_extr": target_cam_trasnf_th[None].float(),  # (1, ?, 4, 4)
        "master_id": master_id[None].long(),  # (1, )
        "master_serial": [master_serial],
    }
    return batch


def collate_batch(batch_list):
    """
    Simple collate function to merge multiple batches using torch.cat
    """
    if not batch_list:
        return None

    if len(batch_list) == 1:
        return batch_list[0]

    collated_batch = {}

    collated_batch["image"] = torch.cat([batch["image"] for batch in batch_list], dim=0)

    collated_batch["target_cam_intr"] = torch.cat([batch["target_cam_intr"] for batch in batch_list], dim=0)
    collated_batch["target_cam_extr"] = torch.cat([batch["target_cam_extr"] for batch in batch_list], dim=0)

    collated_batch["cam_view_num"] = np.concatenate([batch["cam_view_num"] for batch in batch_list], axis=0)
    collated_batch["cam_serial"] = [batch["cam_serial"][0] for batch in batch_list]
    collated_batch["master_id"] = torch.cat([batch["master_id"] for batch in batch_list], dim=0)
    collated_batch["master_serial"] = [batch["master_serial"][0] for batch in batch_list]

    return collated_batch


def extract_pred(pred, batch, req_flip, cam_extr_map):

    def flip_3d(annot_3d):
        annot_3d = annot_3d.copy()
        annot_3d[:, 0] = -annot_3d[:, 0]
        return annot_3d

    # for k, v in pred.items():
    #     print(k, v.shape)
    master_id = batch["master_id"][0]
    master_serial = batch["master_serial"][0]
    joint3d_in_master = pred["pred_joints_3d"][0]
    joint3d_in_master_np = joint3d_in_master.detach().cpu().numpy()
    master_cam_extr = cam_extr_map[master_serial]
    if req_flip:
        master_cam_extr = inv_transf_np(flip_cam_extr(inv_transf_np(master_cam_extr)))
    master_cam_transf = inv_transf_np(master_cam_extr)
    joint3d_in_world_np = transf_point_array_np(master_cam_transf, joint3d_in_master_np)
    if req_flip:
        joint3d_in_world_np = flip_3d(joint3d_in_world_np)
    return {
        "joints": joint3d_in_world_np,
    }


def main(
    cfg: CN,
    arg: Namespace,
    time_f: float,
    cmd_channel: str,
    video_shape: Tuple[int, int],
    sync_channel: str,
    sub_channel: str,
    pub_channel: str,
    identity: str = "poem-0",
):
    logger.info("poem-v2 server start")

    # init socket
    from server_tool import zmq_channel_util

    def parse_channel(channel: str) -> str:
        endpoint = zmq_channel_util.channel_name_to_endpoint(channel, "/dev/shm/hcc_demo")
        if zmq_channel_util.is_ipc_endpoint(endpoint):
            os.makedirs(os.path.dirname(zmq_channel_util.ipc_to_filepath(endpoint)), exist_ok=True)
        return endpoint

    ctx = zmq.Context()

    # Command socket (DEALER) for async communication with master node
    cmd_socket = ctx.socket(zmq.DEALER)
    cmd_socket.setsockopt(zmq.IDENTITY, identity.encode())
    cmd_endpoint = parse_channel(cmd_channel)
    cmd_socket.connect(cmd_endpoint)
    logger.info(f"Command socket (DEALER) connected to: {cmd_endpoint} with identity: {identity}")

    # Report starting status
    cmd_socket.send_string(json.dumps({"status": "starting", "msg": "POEM server starting, loading model..."}))
    logger.info("Reported 'starting' status to master node")

    # Load model FIRST (this is time-consuming but doesn't depend on config)
    logger.info("Loading model (this may take a while)...")
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")
    device = torch.device(f"cuda:0")
    model: ModelABC = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=None, log_freq=arg.log_freq)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Report model_loaded status and wait for init command with config
    cmd_socket.send_string(json.dumps({"status": "model_loaded", "msg": "Model loaded, waiting for config..."}))
    logger.info("Reported 'model_loaded' status, waiting for init command...")

    # Wait for init command with camera_info and calib_filedir from master node
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    camera_info = None
    calib_filedir = None

    while camera_info is None:
        try:
            socks = dict(poller.poll(timeout=100))  # 100ms poll
            if cmd_socket in socks:
                cmd_msg = cmd_socket.recv_string()
                cmd_data = json.loads(cmd_msg)
                if cmd_data.get("cmd") == "init":
                    camera_info = cmd_data.get("camera_info", {})
                    calib_data = cmd_data.get("calib_data", None)
                    calib_filedir = cmd_data.get("calib_filedir", "")
                    logger.info(f"Received camera_info: {camera_info}")
                    if calib_data is not None:
                        logger.info(f"Received calib_data (base64) for {len(calib_data)} cameras")
                    else:
                        logger.info(f"Received calib_filedir: {calib_filedir}")
                elif cmd_data.get("cmd") == "ping":
                    cmd_socket.send_string(json.dumps({"status": "pong", "msg": "waiting for init"}))
                else:
                    logger.warning(f"Unknown command while waiting for init: {cmd_data.get('cmd')}")
        except Exception as e:
            logger.error(f"Error receiving init command: {e}")

    # Load calib based on received camera_info
    camera_name_list = list(camera_info.values())
    cam_extr_map, cam_intr_map = {}, {}

    if calib_data is not None:
        # New protocol: calib data transferred inline via base64-encoded pkl bytes.
        # Works for both local and remote POEM deployment.
        logger.info(f"Loading calib from calib_data (base64) for cameras: {camera_name_list}")
        for cam_name in camera_name_list:
            entry = calib_data[cam_name]
            cam_extr_map[cam_name] = pickle.loads(base64.b64decode(entry["extr"]))
            cam_intr_map[cam_name] = pickle.loads(base64.b64decode(entry["intr"]))
    else:
        # Legacy protocol: load from filesystem using calib_filedir path.
        logger.info(f"Loading calib from {calib_filedir} for cameras: {camera_name_list}")
        for cam_name in camera_name_list:
            extr_filepath = os.path.join(calib_filedir, "cam_extr", f"{cam_name}.pkl")
            with open(extr_filepath, "rb") as ifs:
                cam_extr_map[cam_name] = pickle.load(ifs)
            intr_filepath = os.path.join(calib_filedir, "cam_intr", f"{cam_name}.pkl")
            with open(intr_filepath, "rb") as ifs:
                cam_intr_map[cam_name] = pickle.load(ifs)
    logger.info(f"Loaded calib for {len(camera_name_list)} cameras")

    # Subscribe to synchronized images
    sync_socket = ctx.socket(zmq.SUB)
    sync_socket.setsockopt(zmq.SUBSCRIBE, b"")
    sync_socket.setsockopt(zmq.RCVHWM, 1)
    sync_socket.setsockopt(zmq.CONFLATE, 1)
    sync_channel_endpoint = parse_channel(sync_channel)
    sync_socket.connect(sync_channel_endpoint)
    logger.info(f"Subscribed to sync channel: {sync_channel_endpoint}")

    # Subscribe to WiLoR results (bbox info)
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
    sub_socket.setsockopt(zmq.RCVHWM, 1)
    sub_socket.setsockopt(zmq.CONFLATE, 1)
    sub_channel_endpoint = parse_channel(sub_channel)
    sub_socket.connect(sub_channel_endpoint)
    logger.info(f"Subscribed to sub channel (WiLoR): {sub_channel_endpoint}")

    # Publish 3D joint results
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1)
    pub_socket.setsockopt(zmq.CONFLATE, 1)
    pub_endpoint = parse_channel(pub_channel)
    pub_socket.bind(pub_endpoint)
    logger.info(f"Publishing to: {pub_endpoint}")

    # Report ready status
    cmd_socket.send_string(json.dumps({"status": "ready", "msg": "POEM server ready"}))
    logger.info("Reported 'ready' status to master node")

    # start server
    logger.info("poem server loop")

    # TimestampBuffer for sync frames (only stores img_list + processed flag)
    frame_buffer: TimestampBuffer[SyncFrameCache] = TimestampBuffer(
        max_frame=60,  # 60 frames buffer
        match_threshold=0.1,  # 100ms tolerance
    )

    # Start sync receiver thread
    sync_receiver = SyncReceiverThread(
        sync_socket=sync_socket,
        frame_buffer=frame_buffer,
        video_shape=video_shape,
    )
    sync_receiver.start()
    logger.info("Sync receiver thread started")

    def cleanup():
        sync_receiver.stop()
        sync_receiver.join(timeout=1.0)
        cmd_socket.close()
        sync_socket.close()
        sub_socket.close()
        pub_socket.close()
        ctx.term()
        logger.info("poem-v2 server end")

    atexit.register(cleanup)

    # Timing statistics using deque (like WiLoR)
    frame_count = 0
    infer_times = deque(maxlen=100)
    total_times = deque(maxlen=100)
    LOG_INTERVAL = 30  # Output statistics every 30 frames

    while True:
        try:
            # Receive WiLoR results from sub_channel
            try:
                sub_msg = sub_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                sleep(0.001)
                continue

            # Decode WiLoR message
            sub_data = msgpack.unpackb(sub_msg, object_hook=msgpack_numpy.decode, raw=False)
            wilor_ts = sub_data.get("sync_timestamp") or sub_data.get("timestamp")
            bbox_info = sub_data.get("bbox")

            # Start timing for this frame
            frame_start_time = time()

            if bbox_info is None:
                continue

            # Find matching frame in buffer
            result = frame_buffer.get_nearest(wilor_ts)
            if result is None:
                logger.debug(f"No matching frame for wilor_ts={wilor_ts}")
                continue

            matched_ts, frame = result

            # Skip if already processed
            if frame.processed:
                continue

            # Use matched frame's images
            img_list = frame.img_list
            timestamp = matched_ts

            # process
            res = {}
            batch_store = {}
            for hand_side in ["lh", "rh"]:
                hand_bbox_list = bbox_info.get(hand_side, None)

                if hand_bbox_list is None:
                    res[hand_side] = None
                    batch_store[hand_side] = None
                    continue

                # Filter out None entries (cameras where hand was not detected)
                valid_indices = [
                    i for i, bbox in enumerate(hand_bbox_list)
                    if bbox is not None and i < len(img_list) and img_list[i] is not None
                ]

                if len(valid_indices) < 2:
                    # Need at least 2 views for multi-view reconstruction
                    res[hand_side] = None
                    batch_store[hand_side] = None
                    continue

                valid_cam_names = [camera_name_list[i] for i in valid_indices]
                valid_images = [img_list[i] for i in valid_indices]
                valid_bboxes = [hand_bbox_list[i] for i in valid_indices]

                batch = format_batch(img_list=valid_images,
                                     bbox_list=valid_bboxes,
                                     req_flip=(hand_side == "lh"),
                                     camera_name_list=valid_cam_names,
                                     cam_intr_map=cam_intr_map,
                                     cam_extr_map=cam_extr_map,
                                     img_size=video_shape,
                                     output_size=cfg.DATA_PRESET.IMAGE_SIZE,
                                     device=device)
                batch_store[hand_side] = batch

            valid_batches = [b for b in batch_store.values() if b is not None]
            if not valid_batches:
                continue

            # Check if all valid batches have the same cam_view_num
            view_nums = [b["cam_view_num"][0] for b in valid_batches]
            can_collate = len(set(view_nums)) == 1

            # Track inference time
            infer_start_time = time()

            if can_collate:
                # All batches have the same number of views, use collate for batched inference
                batch = collate_batch(valid_batches)
                with torch.no_grad():
                    pred = model(batch, 0, "inference", epoch_idx=0)

                offset = 0
                for hand_side in ["lh", "rh"]:
                    if batch_store[hand_side] is None:
                        res[hand_side] = None
                        continue

                    pred_item = {k: v[offset:offset + 1] for k, v in pred.items()}

                    payload = extract_pred(pred_item,
                                           batch_store[hand_side],
                                           req_flip=(hand_side == "lh"),
                                           cam_extr_map=cam_extr_map)
                    res[hand_side] = payload

                    offset += 1
            else:
                # Different number of views, process each hand separately
                for hand_side in ["lh", "rh"]:
                    if batch_store[hand_side] is None:
                        res[hand_side] = None
                        continue

                    with torch.no_grad():
                        pred = model(batch_store[hand_side], 0, "inference", epoch_idx=0)

                    pred_item = {k: v[0:1] for k, v in pred.items()}

                    payload = extract_pred(pred_item,
                                           batch_store[hand_side],
                                           req_flip=(hand_side == "lh"),
                                           cam_extr_map=cam_extr_map)
                    res[hand_side] = payload

            # Calculate inference time
            infer_time = time() - infer_start_time

            # reformat res and publish (use sync_timestamp for compatibility with camera_view_recon.py)
            pub_msg = {
                "sync_timestamp": timestamp,
                "pose_3d": {
                    "lh": {
                        "joints": res["lh"]["joints"]
                    } if res["lh"] is not None else None,
                    "rh": {
                        "joints": res["rh"]["joints"]
                    } if res["rh"] is not None else None,
                },
            }
            pub_socket.send(msgpack.packb(pub_msg, default=msgpack_numpy.encode))

            # Calculate frame processing time and record to deque
            total_frame_time = time() - frame_start_time
            frame_count += 1
            infer_times.append(infer_time * 1000)  # Convert to ms
            total_times.append(total_frame_time * 1000)  # Convert to ms

            # Log average statistics every LOG_INTERVAL frames
            if frame_count % LOG_INTERVAL == 0:
                avg_infer_time = np.mean(infer_times) if infer_times else 0
                avg_total_time = np.mean(total_times) if total_times else 0
                logger.info(f"Frame {frame_count} | Infer: {infer_time*1000:.1f}ms (avg: {avg_infer_time:.1f}ms) | "
                            f"Total: {total_frame_time*1000:.1f}ms (avg: {avg_total_time:.1f}ms)")

            # Mark as processed after successful send
            frame.processed = True
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping...")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"POEM error: {e}")
            sleep(0.01)

    cleanup()


MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server.cmd_channel",
                        type=str,
                        required=True,
                        help="ZMQ REP channel for receiving config and reporting ready status")
    parser.add_argument("--server.video_shape", type=str, required=True)
    parser.add_argument("--server.sync_channel", type=str, required=True)
    parser.add_argument("--server.sub_channel", type=str, required=True)
    parser.add_argument("--server.pub_channel", type=str, required=True)
    # poem settings

    server_arg, _ = parser.parse_known_args()

    # parse cmd_channel
    cmd_channel = getattr(server_arg, "server.cmd_channel")
    logger.info(f"cmd_channel: {cmd_channel}")

    # parse video_shape
    video_shape_str = getattr(server_arg, "server.video_shape")
    _split = video_shape_str.split("x", 2)
    video_shape = (int(_split[0]), int(_split[1]))

    # parse channel
    sync_channel = getattr(server_arg, "server.sync_channel")
    sub_channel = getattr(server_arg, "server.sub_channel")
    pub_channel = getattr(server_arg, "server.pub_channel")

    # poem
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # server
    main(cfg=cfg,
         arg=arg,
         time_f=exp_time,
         cmd_channel=cmd_channel,
         video_shape=video_shape,
         sync_channel=sync_channel,
         sub_channel=sub_channel,
         pub_channel=pub_channel)
