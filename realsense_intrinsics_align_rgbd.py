#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not found. Install with: pip install pyrealsense2")
    sys.exit(1)


def intrinsics_to_dict(intr: rs.intrinsics):
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def print_camera_params(profile: rs.pipeline_profile):
    # 获取流 profile
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    # 外参：depth -> color
    depth_to_color_extr = depth_stream.get_extrinsics_to(color_stream)

    print("===== RealSense Camera Parameters =====")
    print("[Color Intrinsics]")
    print(intrinsics_to_dict(color_intr))
    print("\n[Depth Intrinsics]")
    print(intrinsics_to_dict(depth_intr))

    print("\n[Extrinsics: Depth -> Color]")
    print("Rotation (row-major 3x3):")
    r = np.array(depth_to_color_extr.rotation).reshape(3, 3)
    print(r)
    print("Translation (meters):")
    t = np.array(depth_to_color_extr.translation).reshape(3,)
    print(t)

    # depth scale（把 depth units 转成米）
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("\n[Depth Scale]")
    print(f"depth_scale = {depth_scale} (meters per depth unit)")
    print("=======================================")

    return color_intr, depth_intr, depth_scale


def main(
    width=640,
    height=480,
    fps=30,
    save=False,
    out_prefix="realsense_aligned",
    warmup_frames=30,
):
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用流（可按需改成 1280x720 等）
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # 打印内参/外参/深度尺度
    color_intr, depth_intr, depth_scale = print_camera_params(profile)

    # 对齐：把 depth 对齐到 color
    align = rs.align(rs.stream.color)

    # 稳定曝光/自动白平衡
    for _ in range(warmup_frames):
        pipeline.wait_for_frames()

    print("\nPress 'q' to quit. Press 's' to save current aligned RGB-D.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue

            # 转 numpy
            color_np = np.asanyarray(color_frame.get_data())  # BGR uint8
            depth_u16 = np.asanyarray(aligned_depth_frame.get_data())  # uint16 depth units
            depth_m = depth_u16.astype(np.float32) * depth_scale  # meters

            # 可视化深度（仅显示用，不影响 depth_m）
            depth_vis = cv2.convertScaleAbs(depth_m, alpha=255.0 / 2.0)  # 显示到 0~2m
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            vis = np.hstack([color_np, depth_vis])
            cv2.imshow("Color (left) | Aligned Depth (right)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") or save:
                ts = int(time.time() * 1000)
                color_path = f"{out_prefix}_color_{ts}.png"
                depth_png_path = f"{out_prefix}_aligned_depth_u16_{ts}.png"
                depth_npy_path = f"{out_prefix}_aligned_depth_m_{ts}.npy"
                K_path = f"{out_prefix}_K_color_{ts}.txt"

                cv2.imwrite(color_path, color_np)
                # 保存对齐后的 depth 原始 u16（无损），以及 meters 的 npy
                cv2.imwrite(depth_png_path, depth_u16)
                np.save(depth_npy_path, depth_m)

                # 保存彩色相机 K（对齐后的深度与彩色同像素坐标系，用 color_intr 更常用）
                K = np.array([[color_intr.fx, 0, color_intr.ppx],
                              [0, color_intr.fy, color_intr.ppy],
                              [0, 0, 1]], dtype=np.float32)
                np.savetxt(K_path, K, fmt="%.6f")

                print(f"Saved:\n  {color_path}\n  {depth_png_path}\n  {depth_npy_path}\n  {K_path}")
                save = False  # 如果是按键触发，仅保存一次

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 直接运行：python realsense_intrinsics_align_rgbd.py
    # 需要保存：运行后按 's'，或把 save=True
    main(width=640, height=480, fps=30, save=False)
