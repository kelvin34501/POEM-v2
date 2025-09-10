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
import json
import yaml
import pickle
from typing import Dict, Tuple, Optional
import zmq
import atexit
import threading

from server_tool.zmq_msg import decode_sync_message
from server_tool.np_serialize import encode_numpy_json, decode_numpy_json
import queue

import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from lib.utils.transform import _affine_transform, _affine_transform_post_rot

from transform.transform_jit import inv_transf, transf_point_array
from transform.transform_np import inv_transf_np, transf_point_array_np, project_point_array_np

from tool.flip_util import flip_cam_extr


def update_bbox_from_joints(res, camera_name_list, cam_intr_map, cam_extr_map, video_shape):
    """
    Update bounding box information based on predicted 3D joint positions.
    
    Args:
        res: Dictionary containing prediction results for each hand side ("lh", "rh")
        camera_name_list: List of camera names
        cam_intr_map: Dictionary mapping camera names to intrinsic matrices
        cam_extr_map: Dictionary mapping camera names to extrinsic matrices
        video_shape: Tuple of (width, height) for the video resolution
    
    Returns:
        bbox_info_new: Dictionary containing updated bounding box information
    """
    bbox_info_new = {}
    for hand_side in ["lh", "rh"]:
        if res[hand_side] is not None and res[hand_side]["joints"] is not None:
            # 获取世界坐标系下的3D关节点
            joints_3d_world = res[hand_side]["joints"]  # shape: (21, 3)
            bbox_info_new[hand_side] = []

            # 为每个相机计算新的bbox
            for cam_name in camera_name_list:
                cam_intr = cam_intr_map[cam_name]
                cam_extr = cam_extr_map[cam_name]

                # 将世界坐标系下的3D点转换到相机坐标系
                joints_3d_cam = transf_point_array_np(cam_extr, joints_3d_world)

                # 投影到2D图像平面
                joints_2d = project_point_array_np(cam_intr, joints_3d_cam)  # shape: (21, 2)

                # 计算包围盒
                # 获取有效的2D点（深度大于0）
                valid_mask = joints_3d_cam[:, 2] > 0
                if np.sum(valid_mask) > 0:
                    valid_joints_2d = joints_2d[valid_mask]

                    # 计算bbox边界
                    x_min = np.min(valid_joints_2d[:, 0])
                    x_max = np.max(valid_joints_2d[:, 0])
                    y_min = np.min(valid_joints_2d[:, 1])
                    y_max = np.max(valid_joints_2d[:, 1])

                    # 添加边距并确保bbox在图像范围内
                    margin = 50  # 像素边距
                    x_min = max(0, x_min - margin)
                    y_min = max(0, y_min - margin)
                    x_max = min(video_shape[0] - 1, x_max + margin)  # video_shape[0] is width
                    y_max = min(video_shape[1] - 1, y_max + margin)  # video_shape[1] is height

                    # 确保bbox有最小尺寸
                    min_size = 100
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    if bbox_w < min_size:
                        center_x = (x_min + x_max) / 2
                        x_min = max(0, center_x - min_size / 2)
                        x_max = min(video_shape[0] - 1, center_x + min_size / 2)
                    if bbox_h < min_size:
                        center_y = (y_min + y_max) / 2
                        y_min = max(0, center_y - min_size / 2)
                        y_max = min(video_shape[1] - 1, center_y + min_size / 2)

                    bbox_info_new[hand_side].append([x_min, y_min, x_max, y_max])
                else:
                    # 如果没有有效点，使用None
                    bbox_info_new[hand_side].append(None)
        else:
            bbox_info_new[hand_side] = None

    return bbox_info_new


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
    image_th = torch.stack(image_list, axis=0).to(device)
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


def control_thread(
    stop_event,
    cmd_socket,
    status_dict,
    status_lock,
):
    while not stop_event.is_set():
        try:
            msg = cmd_socket.recv_string(zmq.NOBLOCK)
            msg = decode_numpy_json(msg)

            # update the status_dict
            with status_lock:
                status_dict["status"] = msg["status"]
                status_dict["bbox_info"] = msg["bbox_info"]

            recv_ts = msg["timestamp"]

            # just wait until status_dict's ts goes beyond
            while True:
                with status_lock:
                    proc_ts = status_dict["sync_ts"]
                if proc_ts is None or proc_ts >= recv_ts:
                    break

                sleep(0.001)

            # respond with ack
            cmd_socket.send_string("ack")

            sleep(0.001)

        except zmq.Again:
            # No message available - continue
            sleep(0.001)
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"POEM server control error: {e}")


def interpolate_joint3d(joint3d_prev, joint3d_curr, alpha):
    """
    Linear interpolation between two joint3d states.
    
    Args:
        joint3d_prev: Previous joint3d dictionary with 'lh' and 'rh' keys
        joint3d_curr: Current joint3d dictionary with 'lh' and 'rh' keys  
        alpha: Interpolation factor (0.0 = prev, 1.0 = curr)
    
    Returns:
        Interpolated joint3d dictionary
    """
    joint3d_interp = {}
    for hand_side in ["lh", "rh"]:
        prev_joints = joint3d_prev.get(hand_side)
        curr_joints = joint3d_curr.get(hand_side)
        
        if prev_joints is None or curr_joints is None:
            # If either is None, use the non-None one or None if both are None
            joint3d_interp[hand_side] = curr_joints if curr_joints is not None else prev_joints
        else:
            # Linear interpolation
            joint3d_interp[hand_side] = (1 - alpha) * prev_joints + alpha * curr_joints
    
    return joint3d_interp


def sending_thread(
    stop_event,
    pub_socket,
    joint3d_queue,
    send_interval=0.067,  # ~30fps default
):
    """
    Background thread that sends joint3d messages with linear interpolation.
    
    Args:
        stop_event: Threading event to stop the thread
        pub_socket: ZMQ socket for publishing messages
        joint3d_queue: Queue containing joint3d_info dictionaries
        send_interval: Time interval between sends in seconds
    """
    logger.info(f"Starting sending thread with {1.0/send_interval:.1f} fps")
    
    joint3d_prev = None
    joint3d_curr = None
    last_send_time = time()
    interp_counter = 0
    last_sent_timestamp = None  # 跟踪最后发送的时间戳，避免重复
    
    while not stop_event.is_set():
        current_time = time()
        
        # Try to get new joint3d data from queue
        try:
            while True:  # Consume all available messages to get the latest
                joint3d_info = joint3d_queue.get_nowait()
                joint3d_prev = joint3d_curr
                joint3d_curr = joint3d_info
                joint3d_queue.task_done()
                interp_counter = 0  # Reset interpolation counter when getting new data
        except queue.Empty:
            pass
        
        # Check if it's time to send
        if current_time - last_send_time >= send_interval:
            if joint3d_curr is not None:
                # Always send interpolated data based on time progression
                # This ensures monotonic timestamps
                if joint3d_prev is not None:
                    interp_counter += 1
                    # Stop sending if we've interpolated too many frames without new data
                    if interp_counter > 5:
                        # Skip sending but update last_send_time to maintain timing
                        last_send_time = current_time
                    else:
                        # Calculate alpha based on time progression to ensure interpolation (not extrapolation)
                        # Alpha should be between 0 and 1 for true interpolation
                        time_diff = joint3d_curr["timestamp"] - joint3d_prev["timestamp"]
                        if time_diff > 0:
                            # Calculate the expected time for this interpolated frame
                            elapsed_since_prev = interp_counter * send_interval * 1000  # convert to ms
                            alpha = min(elapsed_since_prev / time_diff, 1.0)  # Clamp to [0, 1]
                        else:
                            # If timestamps are the same, use a small fixed interpolation
                            alpha = min(interp_counter * 0.1, 0.5)
                            print(alpha)
                        
                        joint3d_interp = interpolate_joint3d(
                            joint3d_prev["joint3d"], 
                            joint3d_curr["joint3d"], 
                            alpha
                        )
                        
                        # Interpolate timestamp as well (keep as integer)
                        interp_timestamp = int((1 - alpha) * joint3d_prev["timestamp"] + alpha * joint3d_curr["timestamp"])
                        
                        # Send interpolated frame (always mark as interpolated when we have prev data)
                        interp_msg = {
                            "timestamp": interp_timestamp,
                            "joint3d": joint3d_interp,
                            "bbox": joint3d_curr["bbox"],
                            "interpolated": True,
                            "interp_alpha": alpha
                        }
                        interp_msg_json = encode_numpy_json(interp_msg)
                        pub_socket.send_string(interp_msg_json)
                        last_send_time = current_time
                else:
                    # First frame - send as is
                    joint3d_to_send = joint3d_curr.copy()
                    joint3d_to_send["interpolated"] = False
                    send_timestamp = int(joint3d_curr["timestamp"])
                    
                    # 确保时间戳不重复
                    if last_sent_timestamp is not None and send_timestamp <= last_sent_timestamp:
                        send_timestamp = last_sent_timestamp + 1
                    
                    joint3d_to_send["timestamp"] = send_timestamp
                    pub_msg_json = encode_numpy_json(joint3d_to_send)
                    pub_socket.send_string(pub_msg_json)
                    last_send_time = current_time
                    last_sent_timestamp = send_timestamp
            
        sleep(0.001)  # Small sleep to prevent busy waiting
    
    logger.info("Sending thread stopped")


def main(
    cfg: CN,
    arg: Namespace,
    time_f: float,
    camera_info: Dict[str, str],
    cam_extr_map: Optional[Dict[str, np.ndarray]],
    cam_intr_map: Optional[Dict[str, np.ndarray]],
    video_shape: Tuple[int, int],
    sync_channel: str,
    cmd_channel: str,
    pub_channel: str,
    send_fps: float = 15.0,
):
    logger.info("poem-v2 server start")

    # init socket
    from server_tool import zmq_channel_util

    def parse_channel(channel: str) -> zmq.Socket:
        endpoint = zmq_channel_util.channel_name_to_endpoint(channel, "/dev/shm/hcc_demo")
        if zmq_channel_util.is_ipc_endpoint(endpoint):
            os.makedirs(os.path.dirname(zmq_channel_util.ipc_to_filepath(endpoint)), exist_ok=True)
        return endpoint

    ctx = zmq.Context()
    sync_socket = ctx.socket(zmq.SUB)
    sync_socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all messages
    sync_socket.setsockopt(zmq.RCVHWM, 1)
    sync_socket.setsockopt(zmq.CONFLATE, 1)
    sync_channel_endpoint = parse_channel(sync_channel)
    sync_socket.connect(sync_channel_endpoint)

    cmd_socket = ctx.socket(zmq.REP)
    cmd_socket.setsockopt(zmq.RCVTIMEO, 200)
    cmd_endpoint = parse_channel(cmd_channel)
    cmd_socket.bind(cmd_endpoint)

    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1)
    pub_socket.setsockopt(zmq.CONFLATE, 1)
    pub_endpoint = parse_channel(pub_channel)
    pub_socket.bind(pub_endpoint)

    def cleanup():
        sync_socket.close()
        cmd_socket.close()
        pub_socket.close()
        ctx.term()

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
    # Prepare image shapes and dtypes for decoding
    image_shapes = []
    image_dtypes = []
    for camera_name in camera_name_list:
        image_shapes.append((video_shape[1], video_shape[0], 3))  # Color
        image_dtypes.append(np.uint8)
        image_shapes.append((video_shape[1], video_shape[0]))  # Depth
        image_dtypes.append(np.float32)

    # Create queue for joint3d messages
    joint3d_queue = queue.Queue(maxsize=20)  # Limit queue size to prevent memory buildup
    
    # process thread
    thread_stop_event = threading.Event()
    status_dict = {'status': 'idle', 'bbox_info': None, 'sync_ts': None}
    status_lock = threading.Lock()
    control_thread_handle = threading.Thread(target=control_thread,
                                             args=(thread_stop_event, cmd_socket, status_dict, status_lock))
    control_thread_handle.daemon = True
    control_thread_handle.start()
    
    # Start sending thread  
    send_interval = 1.0 / send_fps
    sending_thread_handle = threading.Thread(target=sending_thread,
                                           args=(thread_stop_event, pub_socket, joint3d_queue, send_interval))
    sending_thread_handle.daemon = True
    sending_thread_handle.start()

    # start server
    last_timestamp = None
    logger.info("poem server loop")
    while True:
        if last_timestamp is not None:
            with status_lock:
                status_dict['sync_ts'] = last_timestamp  # so control thread will reply only when bbox is in effect

        with status_lock:
            status_cur = status_dict.copy()
        if status_cur['status'] == "idle":
            sleep(0.001)
            continue
        elif status_cur['status'] == "term":
            break

        try:
            # Receive synchronized images (non-blocking)
            msg = sync_socket.recv(zmq.NOBLOCK)
            images, timestamp = decode_sync_message(msg, image_shapes, image_dtypes)

            if images is None:
                continue

            # Extract RGB images (skip depth images)
            rgb_images = []
            for i in range(0, len(images), 2):  # Every even index is RGB
                rgb_images.append(images[i])

            bbox_info = status_cur['bbox_info']

            # process
            res = {}
            batch_store = {}
            for hand_side in ["lh", "rh"]:
                if bbox_info.get(hand_side, None) is None:
                    res[hand_side] = None

                batch = format_batch(img_list=rgb_images,
                                     bbox_list=bbox_info[hand_side],
                                     req_flip=(hand_side == "lh"),
                                     camera_name_list=camera_name_list,
                                     cam_intr_map=cam_intr_map,
                                     cam_extr_map=cam_extr_map,
                                     img_size=video_shape,
                                     output_size=cfg.DATA_PRESET.IMAGE_SIZE,
                                     device=device)
                batch_store[hand_side] = batch

            valid_batches = [b for b in batch_store.values() if b is not None]
            batch = collate_batch(valid_batches)
            with torch.no_grad():
                pred = model(batch, 0, "inference", epoch_idx=0)

            offset = 0
            for hand_side in ["lh", "rh"]:
                if batch_store[hand_side] is None:
                    res[hand_side] = None

                pred_item = {k: v[offset:offset + 1] for k, v in pred.items()}

                payload = extract_pred(pred_item,
                                       batch_store[hand_side],
                                       req_flip=(hand_side == "lh"),
                                       cam_extr_map=cam_extr_map)
                res[hand_side] = payload

                offset += 1

            # reformat res and add to queue for sending thread
            joint3d_info = {
                "timestamp": timestamp,
                "joint3d": {
                    "lh": res["lh"]["joints"] if res["lh"] is not None else None,
                    "rh": res["rh"]["joints"] if res["rh"] is not None else None,
                },
                "bbox": bbox_info,
            }
            
            # Add to queue for sending thread (non-blocking)
            try:
                joint3d_queue.put_nowait(joint3d_info)
            except queue.Full:
                # If queue is full, remove oldest item and add new one
                try:
                    joint3d_queue.get_nowait()
                    joint3d_queue.put_nowait(joint3d_info)
                except queue.Empty:
                    joint3d_queue.put_nowait(joint3d_info)

            # update bbox info
            bbox_info_new = update_bbox_from_joints(res, camera_name_list, cam_intr_map, cam_extr_map, video_shape)

            # 更新status_dict中的bbox_info，用于下一帧的处理
            with status_lock:
                status_dict["bbox_info"] = bbox_info_new

            # 更新last_timestamp，确保sync_ts能够正确更新
            last_timestamp = timestamp

        except zmq.Again:
            # No message available - continue
            continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"POEM error: {e}")

        # Small sleep to prevent busy waiting
        sleep(0.001)

    thread_stop_event.set()
    control_thread_handle.join()
    sending_thread_handle.join()

    cleanup()
    logger.info("poem-v2 server end")


MODEL_CATEGORY = ['small', 'medium', 'large', 'huge', 'medium_MANO']
EMBED_SIZE = [128, 256, 512, 1024, 256]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--server.camera_info_filepath", type=str, required=True)
    parser.add_argument("--server.calib_filedir", type=str, required=True)
    parser.add_argument("--server.video_shape", type=str, required=True)
    parser.add_argument("--server.sync_channel", type=str, required=True)
    parser.add_argument("--server.cmd_channel", type=str, required=True)
    parser.add_argument("--server.pub_channel", type=str, required=True)
    parser.add_argument("--server.send_fps", type=float, default=15.0, help="Target FPS for sending interpolated joint3d messages")
    # poem settings

    server_arg, _ = parser.parse_known_args()

    # parse camera_info
    camera_info_filepath = getattr(server_arg, "server.camera_info_filepath")
    logger.info(f"load camera info from {camera_info_filepath}")
    with open(camera_info_filepath, "r") as ifs:
        camera_info = yaml.load(ifs, Loader=yaml.SafeLoader)
    camera_info = camera_info['camera_info']  # index to key
    camera_name_list = list(camera_info.values())

    # load calib
    calib_filedir = getattr(server_arg, "server.calib_filedir")
    logger.info(f"load calib from {calib_filedir}")
    cam_extr_map, cam_intr_map = {}, {}
    for cam_name in camera_name_list:
        extr_filepath = os.path.join(calib_filedir, "cam_extr", f"{cam_name}.pkl")
        with open(extr_filepath, "rb") as ifs:
            cam_extr_map[cam_name] = pickle.load(ifs)
        intr_filepath = os.path.join(calib_filedir, "cam_intr", f"{cam_name}.pkl")
        with open(intr_filepath, "rb") as ifs:
            cam_intr_map[cam_name] = pickle.load(ifs)

    # parse video_shape
    video_shape_str = getattr(server_arg, "server.video_shape")
    _split = video_shape_str.split("x", 2)
    video_shape = (int(_split[0]), int(_split[1]))

    # parse channel
    sync_channel = getattr(server_arg, "server.sync_channel")
    cmd_channel = getattr(server_arg, "server.cmd_channel")
    pub_channel = getattr(server_arg, "server.pub_channel")
    send_fps = getattr(server_arg, "server.send_fps", 15.0)

    # poem
    exp_time = time()
    arg, _ = parse_exp_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu_id
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    # server
    main(cfg=cfg,
         arg=arg,
         time_f=exp_time,
         camera_info=camera_info,
         cam_extr_map=cam_extr_map,
         cam_intr_map=cam_intr_map,
         video_shape=video_shape,
         sync_channel=sync_channel,
         cmd_channel=cmd_channel,
         pub_channel=pub_channel,
         send_fps=send_fps)
