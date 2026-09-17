"""Small viewer helpers shared by the public inference demos."""

import numpy as np
import torch
from isaacgym import gymapi

from aerial_gym.utils.math import quat_rotate


def configure_viewer_follow(env, enable_follow, follow_type, env_idx, offset):
    ige_env = getattr(env.sim_env, "IGE_env", None)
    if ige_env is None:
        return
    viewer_ctrl = getattr(ige_env, "viewer", None)
    if viewer_ctrl is None or viewer_ctrl.viewer is None:
        return

    viewer_ctrl.current_target_env = max(
        0, min(env_idx, env.task_config.num_envs - 1)
    )
    viewer_ctrl.camera_follow = enable_follow
    if follow_type == "position":
        viewer_ctrl.camera_follow_type = gymapi.FOLLOW_POSITION
        viewer_ctrl.camera_follow_position_global_offset = torch.tensor(
            offset, device=viewer_ctrl.device, dtype=torch.float32
        )
    else:
        viewer_ctrl.camera_follow_type = gymapi.FOLLOW_TRANSFORM
        viewer_ctrl.camera_follow_transform_local_offset = torch.tensor(
            offset, device=viewer_ctrl.device, dtype=torch.float32
        )
    viewer_ctrl.set_camera_lookat()


def draw_guidance_overlay(
    env,
    actions,
    env_idx=0,
    los_color=(1.0, 0.2, 0.2),
    cmd_color=(0.1, 0.9, 0.9),
    arrow_length=1.5,
    arrow_head_length=0.35,
    arrow_head_width=0.18,
):
    ige_env = getattr(env.sim_env, "IGE_env", None)
    if ige_env is None or ige_env.viewer is None:
        return
    viewer_ctrl = ige_env.viewer
    viewer = getattr(viewer_ctrl, "viewer", None)
    if viewer is None or env_idx >= len(viewer_ctrl.env_handles):
        return

    gym = ige_env.gym
    env_handle = viewer_ctrl.env_handles[env_idx]
    robot_pos = env.obs_dict["robot_position"][env_idx]
    target_pos = env.target_position[env_idx]
    vehicle_quat = env.obs_dict["robot_vehicle_orientation"][env_idx : env_idx + 1]
    cmd_world = quat_rotate(
        vehicle_quat, actions[env_idx, 0:3].unsqueeze(0)
    ).squeeze(0)

    robot_np = robot_pos.detach().cpu().numpy().astype(np.float32)
    target_np = target_pos.detach().cpu().numpy().astype(np.float32)
    vertices = [
        [
            robot_np[0],
            robot_np[1],
            robot_np[2],
            target_np[0],
            target_np[1],
            target_np[2],
        ]
    ]
    colors = [list(los_color)]

    cmd_norm = torch.linalg.vector_norm(cmd_world).item()
    if cmd_norm > 1e-6:
        direction = (cmd_world / cmd_norm).detach().cpu().numpy().astype(np.float32)
        tip = robot_np + arrow_length * direction
        vertices.append([*robot_np, *tip])
        colors.append(list(cmd_color))

        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(direction, up)
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(
                direction, np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
        side /= max(np.linalg.norm(side), 1e-6)
        head_base = tip - arrow_head_length * direction
        for wing in (
            head_base + arrow_head_width * side,
            head_base - arrow_head_width * side,
        ):
            vertices.append([*tip, *wing])
            colors.append(list(cmd_color))

    gym.clear_lines(viewer)
    gym.add_lines(
        viewer,
        env_handle,
        len(vertices),
        np.asarray(vertices, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
    )
