"""
Visualization/inference script for the PNG vy/vz residual intercept policy.

Modes:
  - prior:    run privileged PNG vy/vz prior only
  - residual: run PNG prior + learned vy/vz residual policy
  - compare:  run both sequentially under the same deterministic scene
"""
import script_path_setup  # noqa: F401

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import Optional

os.environ.pop("TORCHINDUCTOR_DISABLE", None)
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import isaacgym  # required import order
import matplotlib.pyplot as plt
import numpy as np
import torch

from viewer_utils import configure_viewer_follow, draw_guidance_overlay
from demo_runtime import (
    PrivilegedPNGVyVzController,
    ResidualVyVzPolicyNet,
    build_env,
    build_full_action_bounds,
    build_policy_obs,
    build_yaw_rate_cmd,
    get_robot_yaw,
    raw_obs_to_policy_obs,
    safe_close_env,
    setup_training_scene,
    LSTMResidualVyPolicy,
)
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg


def configure_external_disturbance(args):
    """Configure the simulator's per-step impulse disturbance for this process."""
    BaseQuadCfg.disturbance.enable_disturbance = bool(args.enable_disturbance)
    BaseQuadCfg.disturbance.prob_apply_disturbance = float(args.disturbance_prob)
    BaseQuadCfg.disturbance.max_force_and_torque_disturbance = [
        float(args.disturbance_force),
        float(args.disturbance_force),
        float(args.disturbance_force),
        float(args.disturbance_torque),
        float(args.disturbance_torque),
        float(args.disturbance_torque),
    ]
    if args.enable_disturbance:
        print(
            "External disturbance enabled: "
            f"prob_per_step={args.disturbance_prob}, "
            f"force=+/-{args.disturbance_force} N, "
            f"torque=+/-{args.disturbance_torque} Nm"
        )


def apply_angle_observation_noise(obs, bias, std_rad):
    """Add angle bias and frame jitter to observations only."""
    noisy = obs.clone()
    noisy[:, 0:2] += bias
    if std_rad > 0.0:
        noisy[:, 0:2] += torch.randn_like(noisy[:, 0:2]) * std_rad
    noisy[:, 0] = torch.clamp(noisy[:, 0], -0.5 * np.pi, 0.5 * np.pi)
    noisy[:, 1] = torch.atan2(torch.sin(noisy[:, 1]), torch.cos(noisy[:, 1]))
    return noisy


def load_checkpoint(checkpoint: str):
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(
            "Checkpoint format mismatch. Expected dict with model_state_dict from "
            "the released interception policy."
        )
    return state


def build_prior_from_ckpt(device: torch.device, ckpt_args: dict):
    return PrivilegedPNGVyVzController(
        max_vy=float(ckpt_args.get("max_vy", 4.0)),
        max_vz=float(ckpt_args.get("max_vz", 2.5)),
        width_scale_px=float(ckpt_args.get("width_scale_px", 50.0)),
        n_nav_y=float(ckpt_args.get("n_nav_y", 2.5)),
        n_nav_z=float(ckpt_args.get("n_nav_z", 2.0)),
        k_pos_y=float(ckpt_args.get("k_pos_y", 0.9)),
        k_pos_z=float(ckpt_args.get("k_pos_z", 1.1)),
        lat_range_cap=float(ckpt_args.get("lat_range_cap", 12.0)),
        vert_range_cap=float(ckpt_args.get("vert_range_cap", 10.0)),
        min_closing_speed=float(ckpt_args.get("min_closing_speed", 1.0)),
        teacher_pos_gain=float(ckpt_args.get("teacher_pos_gain", 1.4)),
        teacher_vel_gain=float(ckpt_args.get("teacher_vel_gain", 0.35)),
        teacher_tgo_floor=float(ckpt_args.get("teacher_tgo_floor", 0.35)),
        pure_los_rate_prior=bool(ckpt_args.get("pure_los_rate_prior", False)),
        rate_prior_closing_speed=float(
            ckpt_args.get(
                "rate_prior_closing_speed",
                ckpt_args.get("forward_speed", 8.0),
            )
            or ckpt_args.get("forward_speed", 8.0)
        ),
        gate_vertical_rate_against_pitch=bool(
            ckpt_args.get("gate_vertical_rate_against_pitch", False)
        ),
    )


def build_policy_from_ckpt(checkpoint_state: dict, device: torch.device):
    if checkpoint_state.get("arch") == "lstm":
        net = LSTMResidualVyPolicy(
            obs_dim=int(checkpoint_state.get("obs_dim", 5)),
            hidden_size=int(checkpoint_state.get("hidden_size", 128)),
        ).to(device)
        net.load_state_dict(checkpoint_state["model_state_dict"])
        net.residual_bounds = checkpoint_state["residual_bounds"].to(device)
        net.obs_dim = int(checkpoint_state.get("obs_dim", 5))
        net.eval()
        return net
    obs_dim = int(checkpoint_state.get("obs_dim", 0) or 0)
    if obs_dim <= 0:
        first_weight = checkpoint_state["model_state_dict"].get("backbone.0.weight")
        obs_dim = int(first_weight.shape[1]) if first_weight is not None else 3
    ckpt_args = checkpoint_state.get("args", {})
    if bool(ckpt_args.get("prior_gate_control", False)):
        act_dim = 2
    else:
        act_dim = 1 if bool(ckpt_args.get("pd_height_control", False)) else 2
    net = ResidualVyVzPolicyNet(obs_dim=obs_dim, act_dim=act_dim).to(device)
    net.load_state_dict(checkpoint_state["model_state_dict"])
    net.residual_bounds = checkpoint_state["residual_bounds"].to(device)
    net.obs_dim = obs_dim
    net.eval()
    return net


def resolve_obs_mode(checkpoint_state: dict) -> str:
    obs_mode = checkpoint_state.get("obs_mode")
    if obs_mode:
        return str(obs_mode)
    if checkpoint_state.get("arch") == "lstm":
        return "fixed_frame_los"
    return "legacy_camera_los"


def disable_fov_termination(env) -> None:
    def check_collision_without_fov():
        robot_positions = env.obs_dict["robot_position"]
        distance = torch.norm(env.target_position - robot_positions, dim=1)
        collision = distance < env.collision_distance_threshold

        collision_reward = env.task_config.reward_parameters["collision_reward"] * collision.float()
        env.rewards += collision_reward
        env.terminations = torch.logical_or(env.terminations, collision)

    env.check_collision = check_collision_without_fov


def set_robot_z(env, robot_z: Optional[float], env_ids=None) -> None:
    if robot_z is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.task_config.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    z_value = float(robot_z)
    robot_state = env.obs_dict["robot_state_tensor"]
    robot_state[env_ids, 2] = z_value
    robot_state[env_ids, 7:13] = 0.0

    sim_env = getattr(env, "sim_env", None)
    write_fn = getattr(sim_env, "write_to_sim", None)
    if callable(write_fn):
        write_fn()
    env.obs_dict = env.sim_env.get_obs()


def setup_target_rear_scene(
    env,
    env_ids=None,
    distance: float = 20.0,
    rear_bearing_deg: float = 30.0,
    target_heading_deg: float = 0.0,
    target_dz: float = 0.0,
    heading_relative_to_robot_yaw: bool = True,
    target_min_z: float = 1.0,
    target_max_z: float = 3.0,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.task_config.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    robot_pos = env.obs_dict["robot_position"][env_ids]
    dtype = robot_pos.dtype
    heading_offset = torch.full(
        (env_ids.numel(),),
        float(np.deg2rad(target_heading_deg)),
        device=env.device,
        dtype=dtype,
    )
    if heading_relative_to_robot_yaw:
        heading = get_robot_yaw(env, env_ids).to(dtype=dtype) + heading_offset
    else:
        heading = heading_offset

    rear_bearing = torch.full(
        (env_ids.numel(),),
        float(np.deg2rad(rear_bearing_deg)),
        device=env.device,
        dtype=dtype,
    )
    dist = torch.full((env_ids.numel(),), float(distance), device=env.device, dtype=dtype)

    target_to_robot_angle = heading + np.pi + rear_bearing
    target_to_robot = torch.stack(
        (
            dist * torch.cos(target_to_robot_angle),
            dist * torch.sin(target_to_robot_angle),
            torch.full_like(dist, -float(target_dz)),
        ),
        dim=1,
    )
    target_pos = robot_pos - target_to_robot
    target_pos[:, 2] = torch.clamp(target_pos[:, 2], min=target_min_z, max=target_max_z)
    env.target_position[env_ids] = target_pos

    direction = torch.stack(
        [torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)],
        dim=1,
    )
    env.target_direction[env_ids] = direction

    target_root = env.asset_state_tensor[env_ids, env.target_asset_idx]
    target_root[:, 0:3] = env.target_position[env_ids]
    target_root[:, 3:7] = 0.0
    target_root[:, 6] = 1.0
    target_root[:, 7:10] = env.target_direction[env_ids] * env.target_velocity_tensor
    target_root[:, 10:13] = 0.0


def setup_demo_scene(env, env_ids=None, **scene_kwargs) -> None:
    target_rear_bearing_deg = scene_kwargs.get("target_rear_bearing_deg")
    if target_rear_bearing_deg is None:
        training_scene_kwargs = dict(scene_kwargs)
        training_scene_kwargs.pop("target_rear_bearing_deg", None)
        training_scene_kwargs.pop("target_rear_align_robot_yaw", None)
        training_scene_kwargs.pop("target_rear_heading_relative_to_robot_yaw", None)
        setup_training_scene(env, env_ids=env_ids, **training_scene_kwargs)
        return

    setup_target_rear_scene(
        env,
        env_ids=env_ids,
        distance=float(scene_kwargs["spawn_distance_min"]),
        rear_bearing_deg=float(target_rear_bearing_deg),
        target_heading_deg=float(scene_kwargs["target_heading_min_deg"]),
        target_dz=float(scene_kwargs["spawn_dz_min"]),
        heading_relative_to_robot_yaw=bool(
            scene_kwargs.get(
                "target_rear_heading_relative_to_robot_yaw",
                scene_kwargs.get("target_rear_align_robot_yaw", True),
            )
        ),
        target_min_z=float(scene_kwargs["target_min_z"]),
        target_max_z=float(scene_kwargs["target_max_z"]),
    )


def save_rollout_logs(base_path: str, logs: dict, metadata: dict) -> None:
    if not base_path:
        return

    root, ext = os.path.splitext(base_path)
    if ext.lower() == ".csv":
        csv_path = base_path
        npz_path = root + ".npz"
    elif ext.lower() == ".npz":
        npz_path = base_path
        csv_path = root + ".csv"
    else:
        csv_path = base_path + ".csv"
        npz_path = base_path + ".npz"

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = list(logs.keys())
    rows = zip(*[logs[k] for k in fieldnames])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    np.savez(
        npz_path,
        **{k: np.asarray(v) for k, v in logs.items()},
        metadata=np.array([metadata], dtype=object),
    )
    print(f"Saved rollout logs to {csv_path} and {npz_path}")


def append_path_suffix(path: str, suffix: str) -> str:
    if not path:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{path}{suffix}"


def log_csv_path(base_path: str) -> str:
    root, ext = os.path.splitext(base_path)
    if ext.lower() == ".csv":
        return base_path
    if ext.lower() == ".npz":
        return root + ".csv"
    return base_path + ".csv"


def load_rollout_csv(csv_path: str) -> dict:
    logs = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                logs.setdefault(key, []).append(float(value))
    return {k: np.asarray(v) for k, v in logs.items()}


def run_angle_sweep_subprocess(args, init_bearing_list: list) -> None:
    results = []
    script_path = os.path.abspath(__file__)
    use_target_rear_bearing = (
        args.target_rear_bearing_deg is not None or args.target_rear_bearing_list_deg is not None
    )
    base_cmd = [
        sys.executable,
        script_path,
        "--checkpoint",
        args.checkpoint,
        "--mode",
        args.mode,
        "--headless",
        "--device",
        args.device,
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--print-every",
        str(args.print_every),
        "--init-distance",
        str(args.init_distance),
        "--init-target-dz",
        str(args.init_target_dz),
        "--target-heading-deg",
        str(args.target_heading_deg),
        "--target-min-z",
        str(args.target_min_z),
        "--target-max-z",
        str(args.target_max_z),
        "--cmd-arrow-length",
        str(args.cmd_arrow_length),
    ]

    if args.use_warp:
        base_cmd.append("--use-warp")
    else:
        base_cmd.append("--no-use-warp")
    if args.keep_alive:
        base_cmd.append("--keep-alive")
    if args.deterministic:
        base_cmd.append("--deterministic")
    if not args.target_rear_heading_relative_to_robot_yaw:
        base_cmd.append("--no-target-rear-heading-relative-to-robot-yaw")
    if not args.ignore_fov_limit:
        base_cmd.append("--respect-fov-limit")
    if args.robot_max_velocity is not None:
        base_cmd.extend(["--robot-max-velocity", str(args.robot_max_velocity)])
    if args.robot_z is not None:
        base_cmd.extend(["--robot-z", str(args.robot_z)])
    if args.forward_speed is not None:
        base_cmd.extend(["--forward-speed", str(args.forward_speed)])
    if args.max_vy is not None:
        base_cmd.extend(["--max-vy", str(args.max_vy)])
    if args.max_vz is not None:
        base_cmd.extend(["--max-vz", str(args.max_vz)])
    if args.max_yaw_rate is not None:
        base_cmd.extend(["--max-yaw-rate", str(args.max_yaw_rate)])
    if args.xy_height_hold:
        base_cmd.append("--xy-height-hold")
        base_cmd.extend(["--hold-kp", str(args.hold_kp)])
        base_cmd.extend(["--hold-kd", str(args.hold_kd)])
        base_cmd.extend(["--hold-max-vz", str(args.hold_max_vz)])
        if args.hold_z is not None:
            base_cmd.extend(["--hold-z", str(args.hold_z)])
    if args.enable_angle_noise:
        base_cmd.extend([
            "--enable-angle-noise",
            "--angle-noise-std-deg",
            str(args.angle_noise_std_deg),
            "--angle-bias-max-deg",
            str(args.angle_bias_max_deg),
        ])

    for angle in init_bearing_list:
        suffix = f"_bearing_{angle:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        angle_plot = append_path_suffix(args.save_plot, suffix)
        angle_log = append_path_suffix(args.save_log, suffix)
        bearing_arg = "--target-rear-bearing-deg" if use_target_rear_bearing else "--init-bearing-deg"
        cmd = base_cmd + [
            bearing_arg,
            str(angle),
            "--save-plot",
            angle_plot,
            "--save-log",
            angle_log,
        ]
        label = "target_rear_bearing" if use_target_rear_bearing else "init_bearing"
        print(f"Running {args.mode} rollout at {label}={angle:+.1f} deg")
        subprocess.run(cmd, check=True)

        logs = load_rollout_csv(log_csv_path(angle_log))
        hit_steps = logs["step"][logs["hit"] > 0.5] if "hit" in logs else np.asarray([])
        min_distance = float(np.min(logs["distance"])) if "distance" in logs else float("inf")
        result = {
            "mode": args.mode,
            "steps": int(len(logs.get("step", []))),
            "avg_reward": float(np.mean(logs["reward"])) if "reward" in logs else 0.0,
            "min_distance": min_distance,
            "hit_step": int(hit_steps[0]) if hit_steps.size else None,
            "init_bearing_deg": angle,
            "logs": logs,
        }
        results.append(result)

    plot_angle_sweep(results, args.save_plot)
    print("Angle sweep summary:")
    for result in results:
        print(
            f"  bearing={result['init_bearing_deg']:+.1f} deg, "
            f"steps={result['steps']}, min_distance={result['min_distance']:.3f}, "
            f"hit_step={result['hit_step']}"
        )
    print(f"Saved angle sweep plot to {args.save_plot}")


@torch.no_grad()
def rollout(
    env,
    prior: PrivilegedPNGVyVzController,
    policy: Optional[ResidualVyVzPolicyNet],
    full_action_bounds: torch.Tensor,
    mode: str,
    steps: int,
    deterministic: bool,
    print_every: int,
    keep_alive: bool,
    scene_kwargs: dict,
    ctrl_kwargs: dict,
    obs_mode: str,
    draw_guidance: bool,
    cmd_arrow_length: float,
):
    raw_obs, _ = env.reset()
    set_robot_z(env, scene_kwargs.get("robot_z"))
    setup_demo_scene(env, **scene_kwargs)
    raw_obs = env.compute_observation()
    obs = torch.nan_to_num(raw_obs_to_policy_obs(env, raw_obs, obs_mode=obs_mode), 0.0)
    noise_std_rad = np.deg2rad(float(ctrl_kwargs.get("angle_noise_std_deg", 0.0)))
    bias_max_rad = np.deg2rad(float(ctrl_kwargs.get("angle_bias_max_deg", 0.0)))
    angle_bias = torch.empty((obs.shape[0], 2), device=env.device, dtype=obs.dtype).uniform_(
        -bias_max_rad, bias_max_rad
    )
    obs = apply_angle_observation_noise(obs, angle_bias, noise_std_rad)
    dt = float(env.sim_env.sim_config.sim.dt)
    prev_obs = None
    is_lstm = isinstance(policy, LSTMResidualVyPolicy)
    hidden = policy.init_hidden(obs.shape[0], env.device) if is_lstm else None
    xy_height_hold = bool(ctrl_kwargs.get("xy_height_hold", False))
    hold_z = ctrl_kwargs.get("hold_z")
    if hold_z is None:
        hold_z = float(env.obs_dict["robot_position"][:, 2].mean().item())
    else:
        hold_z = float(hold_z)

    vy_vz_bounds = torch.tensor(
        [float(ctrl_kwargs["max_vy"]), float(ctrl_kwargs["max_vz"])],
        device=env.device,
    )

    logs = {
        "step": [],
        "pitch": [],
        "yaw": [],
        "width": [],
        "pitch_dot": [],
        "yaw_dot": [],
        "distance": [],
        "reward": [],
        "robot_x": [],
        "robot_y": [],
        "robot_z": [],
        "robot_yaw": [],
        "target_x": [],
        "target_y": [],
        "target_z": [],
        "raw_prior_vy": [],
        "scaled_prior_vy": [],
        "prior_vy": [],
        "prior_vz": [],
        "cmd_vx": [],
        "cmd_vy": [],
        "cmd_vz": [],
        "cmd_yaw_rate": [],
        "res_vy": [],
        "res_vz": [],
        "prior_gate": [],
        "gate_logit": [],
        "speed_xy": [],
        "hit": [],
        "terminated": [],
        "truncated": [],
    }

    total_reward = 0.0
    min_distance = float("inf")
    hit_step = None
    hit_threshold = float(ctrl_kwargs["hit_distance"])

    for step_idx in range(steps):
        obs_in = torch.nan_to_num(obs, 0.0)
        policy_obs_full = torch.nan_to_num(build_policy_obs(obs_in, prev_obs, dt), 0.0)
        pitch_dot = policy_obs_full[:, 3]
        yaw_dot = policy_obs_full[:, 4]

        prior_cmd, _ = prior.prior_command(env, policy_obs=policy_obs_full)
        prior_cmd_for_control = prior_cmd.clone()
        if ctrl_kwargs["weighted_prior_control"]:
            prior_cmd_for_control[:, 0] = float(ctrl_kwargs["prior_vy_weight"]) * prior_cmd_for_control[:, 0]
        yaw_rate_cmd = build_yaw_rate_cmd(
            obs_in,
            prev_obs,
            dt,
            k_yaw=float(ctrl_kwargs["yaw_kp"]),
            k_yaw_d=float(ctrl_kwargs["yaw_kd"]),
            max_yaw_rate=float(ctrl_kwargs["max_yaw_rate"]),
        )

        if mode == "prior":
            residual = torch.zeros_like(prior_cmd_for_control)
            vy_vz_cmd = prior_cmd_for_control.clone()
        else:
            if policy is None:
                raise RuntimeError("policy is required for residual mode")
            policy_input = policy_obs_full[:, : int(getattr(policy, "obs_dim", policy_obs_full.shape[1]))]
            if is_lstm:
                if deterministic:
                    mean, _, hidden_next = policy.forward_step(policy_input, hidden)
                    residual = torch.tanh(mean) * policy.residual_bounds
                else:
                    residual, _, _, hidden_next = policy.sample_step(policy_input, hidden)
                hidden = hidden_next
            elif deterministic:
                mean, _, _ = policy(policy_input)
                residual = torch.tanh(mean) * policy.residual_bounds
            else:
                residual, _, _, _ = policy.act(policy_input)
            if ctrl_kwargs["prior_gate_control"]:
                vy_vz_cmd = torch.zeros((prior_cmd_for_control.shape[0], 2), device=env.device, dtype=prior_cmd_for_control.dtype)
                prior_gate = torch.sigmoid(residual[:, 1])
                vy_vz_cmd[:, 0] = torch.clamp(
                    prior_gate * prior_cmd_for_control[:, 0] + residual[:, 0],
                    -float(ctrl_kwargs["max_vy"]),
                    float(ctrl_kwargs["max_vy"]),
                )
            elif ctrl_kwargs["pd_height_control"]:
                vy_vz_cmd = torch.zeros((prior_cmd_for_control.shape[0], 2), device=env.device, dtype=prior_cmd_for_control.dtype)
                vy_vz_cmd[:, 0] = torch.clamp(prior_cmd_for_control[:, 0] + residual[:, 0], -float(ctrl_kwargs["max_vy"]), float(ctrl_kwargs["max_vy"]))
            else:
                if xy_height_hold:
                    residual = residual.clone()
                    residual[:, 1] = 0.0
                vy_vz_cmd = torch.clamp(prior_cmd_for_control + residual, -vy_vz_bounds, vy_vz_bounds)

        if ctrl_kwargs["pd_height_control"] or xy_height_hold:
            robot_z = env.obs_dict["robot_position"][:, 2]
            robot_linvel = env.obs_dict.get("robot_linvel")
            robot_vz = torch.zeros_like(robot_z)
            if robot_linvel is not None:
                robot_vz = robot_linvel[:, 2]
            height_target = (
                env.target_position[:, 2]
                if ctrl_kwargs["pd_height_control"] and not xy_height_hold
                else torch.full_like(robot_z, hold_z)
            )
            hold_vz = (
                float(ctrl_kwargs["hold_kp"]) * (height_target - robot_z)
                - float(ctrl_kwargs["hold_kd"]) * robot_vz
            )
            hold_max_vz = min(float(ctrl_kwargs["hold_max_vz"]), float(ctrl_kwargs["max_vz"]))
            vy_vz_cmd = vy_vz_cmd.clone()
            vy_vz_cmd[:, 1] = torch.clamp(hold_vz, -hold_max_vz, hold_max_vz)

        action = torch.zeros((obs_in.shape[0], 4), device=env.device, dtype=obs_in.dtype)
        action[:, 0] = min(float(ctrl_kwargs["forward_speed"]), float(full_action_bounds[0].item()))
        action[:, 1] = torch.clamp(vy_vz_cmd[:, 0], -float(ctrl_kwargs["max_vy"]), float(ctrl_kwargs["max_vy"]))
        action[:, 2] = torch.clamp(vy_vz_cmd[:, 1], -float(ctrl_kwargs["max_vz"]), float(ctrl_kwargs["max_vz"]))
        action[:, 3] = yaw_rate_cmd

        next_raw_obs, reward, term, trunc, _ = env.step(action)
        next_obs = torch.nan_to_num(raw_obs_to_policy_obs(env, next_raw_obs, obs_mode=obs_mode), 0.0)
        next_obs = apply_angle_observation_noise(next_obs, angle_bias, noise_std_rad)
        reward = reward.squeeze(-1) if reward.dim() > 1 else reward
        distance = torch.norm(env.target_position - env.obs_dict["robot_position"], dim=1)
        hit_mask = distance <= hit_threshold

        total_reward += reward.mean().item()
        min_distance = min(min_distance, distance.min().item())
        if hit_step is None and hit_mask.any():
            hit_step = step_idx + 1

        robot_pos0 = env.obs_dict["robot_position"][0]
        robot_yaw0 = env.obs_dict["robot_euler_angles"][0, 2].item()
        target_pos0 = env.target_position[0]
        robot_linvel = env.obs_dict.get("robot_linvel")
        speed_xy0 = 0.0
        if robot_linvel is not None:
            speed_xy0 = torch.linalg.vector_norm(robot_linvel[0, 0:2], dim=0).item()

        logs["step"].append(step_idx + 1)
        logs["pitch"].append(obs_in[0, 0].item())
        logs["yaw"].append(obs_in[0, 1].item())
        logs["width"].append(obs_in[0, 2].item())
        logs["pitch_dot"].append(pitch_dot[0].item())
        logs["yaw_dot"].append(yaw_dot[0].item())
        logs["distance"].append(distance[0].item())
        logs["reward"].append(reward[0].item())
        logs["robot_x"].append(robot_pos0[0].item())
        logs["robot_y"].append(robot_pos0[1].item())
        logs["robot_z"].append(robot_pos0[2].item())
        logs["robot_yaw"].append(robot_yaw0)
        logs["target_x"].append(target_pos0[0].item())
        logs["target_y"].append(target_pos0[1].item())
        logs["target_z"].append(target_pos0[2].item())
        logs["raw_prior_vy"].append(prior_cmd[0, 0].item())
        logs["scaled_prior_vy"].append(prior_cmd_for_control[0, 0].item())
        logs["prior_vy"].append(prior_cmd_for_control[0, 0].item())
        logs["prior_vz"].append(prior_cmd[0, 1].item())
        logs["cmd_vx"].append(action[0, 0].item())
        logs["cmd_vy"].append(action[0, 1].item())
        logs["cmd_vz"].append(action[0, 2].item())
        logs["cmd_yaw_rate"].append(action[0, 3].item())
        logs["res_vy"].append(residual[0, 0].item())
        logs["res_vz"].append(0.0 if ctrl_kwargs["prior_gate_control"] else (residual[0, 1].item() if residual.shape[1] > 1 else 0.0))
        logs["prior_gate"].append(torch.sigmoid(residual[0, 1]).item() if ctrl_kwargs["prior_gate_control"] else 1.0)
        logs["gate_logit"].append(residual[0, 1].item() if ctrl_kwargs["prior_gate_control"] else 0.0)
        logs["speed_xy"].append(speed_xy0)
        logs["hit"].append(int(hit_mask[0].item()))
        logs["terminated"].append(int(term[0].item()))
        logs["truncated"].append(int(trunc[0].item()))

        if draw_guidance:
            draw_guidance_overlay(env, actions=action, env_idx=0, arrow_length=cmd_arrow_length)

        if (step_idx + 1) % print_every == 0:
            print(
                f"[{mode}] step={step_idx + 1:5d}, avgR={total_reward/max(step_idx + 1, 1):.3f}, "
                f"dist={distance[0].item():.3f}, obs0=[pitch={obs_in[0,0].item():.3f}, "
                f"yaw={obs_in[0,1].item():.3f}, width={obs_in[0,2].item():.2f}], "
                f"cmd0=[vx={action[0,0].item():.2f}, vy={action[0,1].item():.2f}, "
                f"vz={action[0,2].item():.2f}, yaw_rate={action[0,3].item():.2f}]"
            )

        done = term | trunc | hit_mask
        next_prev_obs = obs_in.detach()
        if done.any():
            if not keep_alive:
                break
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            angle_bias[done_ids] = torch.empty(
                (done_ids.numel(), 2), device=env.device, dtype=obs.dtype
            ).uniform_(-bias_max_rad, bias_max_rad)
            env.reset_idx(done_ids)
            set_robot_z(env, scene_kwargs.get("robot_z"), env_ids=done_ids)
            setup_demo_scene(env, env_ids=done_ids, **scene_kwargs)
            next_raw_obs = env.compute_observation()
            next_obs = torch.nan_to_num(raw_obs_to_policy_obs(env, next_raw_obs, obs_mode=obs_mode), 0.0)
            next_obs = apply_angle_observation_noise(next_obs, angle_bias, noise_std_rad)
            next_prev_obs = next_obs.detach()
            if is_lstm:
                hidden = (hidden[0].clone(), hidden[1].clone())
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                hidden[0][:, done_ids] = 0.0
                hidden[1][:, done_ids] = 0.0

        obs = next_obs
        prev_obs = next_prev_obs

    result = {
        "mode": mode,
        "steps": len(logs["step"]),
        "avg_reward": total_reward / max(len(logs["step"]), 1),
        "min_distance": min_distance,
        "hit_step": hit_step,
        "logs": {k: np.asarray(v) for k, v in logs.items()},
    }
    return result


def plot_single(result: dict, save_path: str):
    logs = result["logs"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(logs["step"], logs["distance"], label="distance [m]")
    axes[0].plot(logs["step"], logs["reward"], label="reward")
    axes[0].grid(True)
    axes[0].legend(loc="best")
    axes[0].set_ylabel("Metric")

    axes[1].plot(logs["step"], logs["pitch"], label="pitch [rad]")
    axes[1].plot(logs["step"], logs["yaw"], label="yaw [rad]")
    axes[1].plot(logs["step"], logs["width"], label="width [px]")
    axes[1].grid(True)
    axes[1].legend(loc="best")
    axes[1].set_ylabel("Obs")

    axes[2].plot(logs["step"], logs["prior_vy"], label="prior vy")
    axes[2].plot(logs["step"], logs["prior_vz"], label="prior vz")
    axes[2].plot(logs["step"], logs["cmd_yaw_rate"], label="yaw_rate")
    axes[2].grid(True)
    axes[2].legend(loc="best")
    axes[2].set_ylabel("Prior / Yaw")

    axes[3].plot(logs["step"], logs["cmd_vx"], label="cmd vx")
    axes[3].plot(logs["step"], logs["cmd_vy"], label="cmd vy")
    axes[3].plot(logs["step"], logs["cmd_vz"], label="cmd vz")
    if np.abs(logs["res_vy"]).sum() > 0.0 or np.abs(logs["res_vz"]).sum() > 0.0:
        axes[3].plot(logs["step"], logs["res_vy"], "--", label="res vy")
        axes[3].plot(logs["step"], logs["res_vz"], "--", label="res vz")
    axes[3].grid(True)
    axes[3].legend(loc="best")
    axes[3].set_ylabel("Final Cmd")
    axes[3].set_xlabel("Step")

    fig.suptitle(
        f"{result['mode']} rollout | avg_reward={result['avg_reward']:.3f}, "
        f"min_distance={result['min_distance']:.3f}, hit_step={result['hit_step']}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_compare(prior_result: dict, residual_result: dict, save_path: str):
    p = prior_result["logs"]
    r = residual_result["logs"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(p["step"], p["distance"], label="prior distance")
    axes[0].plot(r["step"], r["distance"], label="residual distance")
    axes[0].grid(True)
    axes[0].legend(loc="best")
    axes[0].set_ylabel("Distance [m]")

    axes[1].plot(p["step"], p["yaw"], label="prior yaw")
    axes[1].plot(r["step"], r["yaw"], label="residual yaw")
    axes[1].plot(p["step"], p["pitch"], label="prior pitch")
    axes[1].plot(r["step"], r["pitch"], label="residual pitch")
    axes[1].grid(True)
    axes[1].legend(loc="best")
    axes[1].set_ylabel("LOS [rad]")

    axes[2].plot(p["step"], p["cmd_vy"], label="prior vy")
    axes[2].plot(r["step"], r["cmd_vy"], label="residual vy")
    axes[2].plot(p["step"], p["cmd_vz"], label="prior vz")
    axes[2].plot(r["step"], r["cmd_vz"], label="residual vz")
    axes[2].grid(True)
    axes[2].legend(loc="best")
    axes[2].set_ylabel("Cmd")

    axes[3].plot(r["step"], r["res_vy"], label="res vy")
    axes[3].plot(r["step"], r["res_vz"], label="res vz")
    axes[3].grid(True)
    axes[3].legend(loc="best")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Step")

    fig.suptitle(
        "PNG vy/vz prior vs prior+residual | "
        f"prior min_dist={prior_result['min_distance']:.3f}, "
        f"residual min_dist={residual_result['min_distance']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_angle_sweep(results: list, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("viridis", max(len(results), 1))

    for idx, result in enumerate(results):
        logs = result["logs"]
        angle = result["init_bearing_deg"]
        color = cmap(idx)
        label = (
            f"{angle:+.1f} deg | min={result['min_distance']:.2f} m | "
            f"hit={result['hit_step']}"
        )
        axes[0].plot(logs["robot_x"], logs["robot_y"], color=color, label=label)
        axes[0].plot(logs["target_x"], logs["target_y"], "--", color=color, alpha=0.55)
        axes[0].scatter(logs["robot_x"][0], logs["robot_y"][0], marker="o", color=color, s=20)
        axes[0].scatter(logs["target_x"][0], logs["target_y"][0], marker="x", color=color, s=35)
        axes[1].plot(logs["step"], logs["distance"], color=color, label=f"{angle:+.1f} deg")

    axes[0].set_title("XY trajectories: solid=UAV, dashed=target")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_title("Distance history")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distance [m]")
    axes[1].grid(True)
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def print_summary(result: dict):
    print(
        f"{result['mode']}: steps={result['steps']}, avg_reward={result['avg_reward']:.3f}, "
        f"min_distance={result['min_distance']:.3f}, hit_step={result['hit_step']}"
    )


def main():
    ap = argparse.ArgumentParser(description="Visualize the PNG vy/vz residual intercept policy.")
    ap.add_argument("--checkpoint", type=str, default="png_vy_vz_residual_policy_ppo.pth")
    ap.add_argument("--mode", choices=["prior", "residual", "compare"], default="residual")
    ap.add_argument("--viewer", action="store_true", help="Enable Isaac Gym viewer for single-env rollout.")
    ap.add_argument("--headless", action="store_true", help="Force headless mode.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--use-warp", action="store_true", default=False)
    ap.add_argument("--no-use-warp", action="store_false", dest="use_warp")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--keep-alive", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--save-plot", type=str, default="png_vy_vz_residual_rollout.png")
    ap.add_argument("--save-log", type=str, default="png_vy_vz_residual_rollout")
    ap.add_argument("--show-plot", action="store_true")

    ap.add_argument("--follow-uav", action="store_true", default=True)
    ap.add_argument("--no-follow-uav", action="store_false", dest="follow_uav")
    ap.add_argument("--follow-type", choices=["transform", "position"], default="transform")
    ap.add_argument("--follow-env", type=int, default=0)
    ap.add_argument("--follow-offset", type=float, nargs=3, default=(-3.0, 0.0, 1.2))
    ap.add_argument("--draw-guidance", action="store_true", default=True)
    ap.add_argument("--no-draw-guidance", action="store_false", dest="draw_guidance")
    ap.add_argument("--cmd-arrow-length", type=float, default=1.5)

    ap.add_argument("--init-distance", type=float, default=20.0)
    ap.add_argument("--init-bearing-deg", type=float, default=15.0)
    ap.add_argument(
        "--init-bearing-list-deg",
        type=float,
        nargs="+",
        default=None,
        help="Run several initial bearing angles sequentially, e.g. -30 -15 0 15 30.",
    )
    ap.add_argument(
        "--target-rear-bearing-deg",
        type=float,
        default=None,
        help="Place the UAV behind the target, at this angle from the target rear axis.",
    )
    ap.add_argument(
        "--target-rear-bearing-list-deg",
        type=float,
        nargs="+",
        default=None,
        help="Run several target-rear bearing angles sequentially, e.g. -30 30.",
    )
    ap.add_argument("--robot-z", type=float, default=None, help="Set initial UAV z height before placing the target.")
    ap.add_argument(
        "--init-target-dz",
        type=float,
        default=0.0,
        help="Initial target height difference in meters; overridden by --init-target-pitch-deg.",
    )
    ap.add_argument(
        "--init-target-pitch-deg",
        type=float,
        default=None,
        help="Initial vertical LOS angle in degrees; dz = horizontal distance * tan(angle).",
    )
    ap.add_argument("--target-heading-deg", type=float, default=0.0)
    ap.add_argument("--target-rear-heading-relative-to-robot-yaw", action="store_true", default=True)
    ap.add_argument(
        "--no-target-rear-heading-relative-to-robot-yaw",
        action="store_false",
        dest="target_rear_heading_relative_to_robot_yaw",
    )
    ap.add_argument(
        "--target-rear-align-robot-yaw",
        action="store_true",
        dest="target_rear_heading_relative_to_robot_yaw",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--no-target-rear-align-robot-yaw",
        action="store_false",
        dest="target_rear_heading_relative_to_robot_yaw",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--target-min-z", type=float, default=1.0)
    ap.add_argument("--target-max-z", type=float, default=3.0)
    ap.add_argument("--robot-max-velocity", type=float, default=None)
    ap.add_argument("--forward-speed", type=float, default=None)
    ap.add_argument("--max-vy", type=float, default=None)
    ap.add_argument("--max-vz", type=float, default=None)
    ap.add_argument("--max-yaw-rate", type=float, default=None)
    ap.add_argument(
        "--xy-height-hold",
        action="store_true",
        help="Ignore PNG/policy vz and use a simple PD height hold command.",
    )
    ap.add_argument("--hold-z", type=float, default=None, help="Fixed hold height in meters. Defaults to initial UAV z.")
    ap.add_argument("--hold-kp", type=float, default=None)
    ap.add_argument("--hold-kd", type=float, default=None)
    ap.add_argument("--hold-max-vz", type=float, default=3.0)
    ap.add_argument("--enable-disturbance", action="store_true")
    ap.add_argument("--disturbance-prob", type=float, default=0.01)
    ap.add_argument("--disturbance-force", type=float, default=0.30)
    ap.add_argument("--disturbance-torque", type=float, default=0.002)
    ap.add_argument("--enable-angle-noise", action="store_true")
    ap.add_argument("--angle-noise-std-deg", type=float, default=0.0)
    ap.add_argument("--angle-bias-max-deg", type=float, default=0.0)
    ap.add_argument("--ignore-fov-limit", action="store_true", default=True)
    ap.add_argument("--respect-fov-limit", action="store_false", dest="ignore_fov_limit")
    args = ap.parse_args()

    configure_external_disturbance(args)
    if args.enable_angle_noise:
        print(f"Angle observation noise enabled: std={args.angle_noise_std_deg:.2f} deg, bias=+/-{args.angle_bias_max_deg:.2f} deg")
    else:
        print("Angle observation noise disabled.")
    if not args.enable_disturbance:
        print("External disturbance disabled.")

    sys.argv = [sys.argv[0]]

    if args.viewer and args.headless:
        args.viewer = False
    if args.viewer and not os.environ.get("DISPLAY"):
        print("DISPLAY not found; forcing headless mode.")
        args.viewer = False
        args.headless = True
    if args.viewer and args.mode == "compare":
        print("Viewer is not supported in compare mode; forcing headless compare rollout.")
        args.viewer = False
        args.headless = True
    if args.viewer and not args.use_warp:
        print("Viewer mode forces --use-warp for safer sensor/viewer interaction.")
        args.use_warp = True

    if args.target_rear_bearing_list_deg is not None:
        init_bearing_list = [float(v) for v in args.target_rear_bearing_list_deg]
    elif args.init_bearing_list_deg is not None:
        init_bearing_list = [float(v) for v in args.init_bearing_list_deg]
    elif args.target_rear_bearing_deg is not None:
        init_bearing_list = [float(args.target_rear_bearing_deg)]
    else:
        init_bearing_list = [float(args.init_bearing_deg)]
    if len(init_bearing_list) > 1 and args.mode == "compare":
        raise ValueError("--init-bearing-list-deg is supported for --mode prior/residual, not compare.")
    if len(init_bearing_list) > 1 and args.viewer:
        print("Multi-angle sweep forces headless mode. Use --init-bearing-deg for one viewer run.")
        args.viewer = False
        args.headless = True

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if len(init_bearing_list) > 1:
        run_angle_sweep_subprocess(args, init_bearing_list)
        return

    ckpt = load_checkpoint(args.checkpoint)
    ckpt_args = ckpt.get("args", {})
    obs_mode = resolve_obs_mode(ckpt)
    print(f"Observation mode: {obs_mode}")
    headless = args.headless or (not args.viewer)

    if args.init_target_pitch_deg is not None:
        initial_target_dz = float(
            args.init_distance * np.tan(np.deg2rad(args.init_target_pitch_deg))
        )
        print(
            f"Initial pitch angle={args.init_target_pitch_deg:+.2f} deg -> "
            f"target dz={initial_target_dz:+.3f} m "
            f"(horizontal distance={args.init_distance:.3f} m)"
        )
    else:
        initial_target_dz = float(args.init_target_dz)

    scene_kwargs = {
        "spawn_distance_min": args.init_distance,
        "spawn_distance_max": args.init_distance,
        "spawn_bearing_min_deg": args.init_bearing_deg,
        "spawn_bearing_max_deg": args.init_bearing_deg,
        "robot_z": args.robot_z,
        "spawn_dz_min": initial_target_dz,
        "spawn_dz_max": initial_target_dz,
        "target_heading_min_deg": args.target_heading_deg,
        "target_heading_max_deg": args.target_heading_deg,
        "target_min_z": args.target_min_z,
        "target_max_z": args.target_max_z,
        "target_rear_bearing_deg": args.target_rear_bearing_deg,
        "target_rear_heading_relative_to_robot_yaw": args.target_rear_heading_relative_to_robot_yaw,
    }

    is_lstm_checkpoint = ckpt.get("arch") == "lstm"
    ctrl_kwargs = {
        "forward_speed": float(args.forward_speed if args.forward_speed is not None else ckpt_args.get("forward_speed", 8.0)),
        "max_vy": float(args.max_vy if args.max_vy is not None else ckpt_args.get("max_vy", 4.0)),
        "max_vz": float(args.max_vz if args.max_vz is not None else ckpt_args.get("max_vz", 2.5)),
        "max_yaw_rate": float(args.max_yaw_rate if args.max_yaw_rate is not None else ckpt_args.get("max_yaw_rate", 1.2)),
        "yaw_kp": float(ckpt_args.get("yaw_kp", 1.2)),
        "yaw_kd": float(ckpt_args.get("yaw_kd", 0.18)),
        "hit_distance": float(ckpt_args.get("hit_distance", 0.5)),
        "pd_height_control": bool(ckpt_args.get("pd_height_control", is_lstm_checkpoint)),
        "prior_gate_control": bool(ckpt_args.get("prior_gate_control", False)),
        "weighted_prior_control": bool(ckpt_args.get("weighted_prior_control", False)),
        "prior_vy_weight": float(ckpt_args.get("prior_vy_weight", 1.0)),
        "xy_height_hold": bool(args.xy_height_hold),
        "hold_z": args.hold_z,
        "hold_kp": float(args.hold_kp if args.hold_kp is not None else ckpt_args.get("height_hold_kp", 2.0)),
        "hold_kd": float(args.hold_kd if args.hold_kd is not None else ckpt_args.get("height_hold_kd", 0.7)),
        "hold_max_vz": float(args.hold_max_vz),
        "angle_noise_std_deg": float(args.angle_noise_std_deg if args.enable_angle_noise else 0.0),
        "angle_bias_max_deg": float(args.angle_bias_max_deg if args.enable_angle_noise else 0.0),
    }
    prior_args = dict(ckpt_args)
    prior_args["max_vy"] = ctrl_kwargs["max_vy"]
    prior_args["max_vz"] = ctrl_kwargs["max_vz"]

    def make_env():
        return build_env(
            num_envs=args.num_envs,
            device=args.device,
            use_warp=args.use_warp,
            robot_name=str(ckpt_args.get("robot_name", "base_quadrotor")),
            controller_name=str(ckpt_args.get("controller_name", "lee_velocity_control")),
            robot_max_velocity=float(args.robot_max_velocity if args.robot_max_velocity is not None else ckpt_args.get("robot_max_velocity", 15.0)),
        target_speed=float(ckpt_args.get("target_speed", 7.0)),
        headless=headless,
        )

    full_action_bounds = build_full_action_bounds(
        device=torch.device(args.device),
        max_velocity=float(args.robot_max_velocity if args.robot_max_velocity is not None else ckpt_args.get("robot_max_velocity", 15.0)),
        max_vy=ctrl_kwargs["max_vy"],
        max_vz=ctrl_kwargs["max_vz"],
        max_yaw_rate=ctrl_kwargs["max_yaw_rate"],
    )

    if args.mode == "compare":
        env_prior = make_env()
        if args.ignore_fov_limit:
            disable_fov_termination(env_prior)
        prior = build_prior_from_ckpt(env_prior.device, prior_args)
        prior_result = rollout(
            env_prior,
            prior=prior,
            policy=None,
            full_action_bounds=full_action_bounds.to(env_prior.device),
            mode="prior",
            steps=args.steps,
            deterministic=True,
            print_every=args.print_every,
            keep_alive=args.keep_alive,
            scene_kwargs=scene_kwargs,
            ctrl_kwargs=ctrl_kwargs,
            obs_mode=obs_mode,
            draw_guidance=False,
            cmd_arrow_length=args.cmd_arrow_length,
        )
        safe_close_env(env_prior)

        env_res = make_env()
        if args.ignore_fov_limit:
            disable_fov_termination(env_res)
        prior = build_prior_from_ckpt(env_res.device, prior_args)
        policy = build_policy_from_ckpt(ckpt, env_res.device)
        residual_result = rollout(
            env_res,
            prior=prior,
            policy=policy,
            full_action_bounds=full_action_bounds.to(env_res.device),
            mode="residual",
            steps=args.steps,
            deterministic=args.deterministic,
            print_every=args.print_every,
            keep_alive=args.keep_alive,
            scene_kwargs=scene_kwargs,
            ctrl_kwargs=ctrl_kwargs,
            obs_mode=obs_mode,
            draw_guidance=False,
            cmd_arrow_length=args.cmd_arrow_length,
        )
        print_summary(prior_result)
        print_summary(residual_result)
        plot_compare(prior_result, residual_result, args.save_plot)
        save_rollout_logs(
            args.save_log + "_prior",
            prior_result["logs"],
            {"mode": "prior", "args": vars(args), "ckpt_args": ckpt_args},
        )
        save_rollout_logs(
            args.save_log + "_residual",
            residual_result["logs"],
            {"mode": "residual", "args": vars(args), "ckpt_args": ckpt_args},
        )
        safe_close_env(env_res)
    else:
        env = make_env()
        if args.ignore_fov_limit:
            disable_fov_termination(env)
            print("FOV lost-target termination disabled for this demo run.")
        if args.xy_height_hold:
            hold_z_text = "initial UAV z" if args.hold_z is None else f"{args.hold_z:.3f} m"
            print(
                "XY height-hold mode enabled: "
                f"policy/PNG vz ignored, hold_z={hold_z_text}, "
                f"kp={args.hold_kp:.2f}, kd={args.hold_kd:.2f}, max_vz={args.hold_max_vz:.2f}"
            )
        if args.viewer:
            configure_viewer_follow(
                env,
                enable_follow=args.follow_uav,
                follow_type=args.follow_type,
                env_idx=args.follow_env,
                offset=args.follow_offset,
            )
        prior = build_prior_from_ckpt(env.device, prior_args)
        policy = build_policy_from_ckpt(ckpt, env.device) if args.mode == "residual" else None
        result = rollout(
            env,
            prior=prior,
            policy=policy,
            full_action_bounds=full_action_bounds.to(env.device),
            mode=args.mode,
            steps=args.steps,
            deterministic=args.deterministic,
            print_every=args.print_every,
            keep_alive=args.keep_alive,
            scene_kwargs=scene_kwargs,
            ctrl_kwargs=ctrl_kwargs,
            obs_mode=obs_mode,
            draw_guidance=args.viewer and args.draw_guidance,
            cmd_arrow_length=args.cmd_arrow_length,
        )
        print_summary(result)
        plot_single(result, args.save_plot)
        save_rollout_logs(
            args.save_log,
            result["logs"],
            {"mode": args.mode, "args": vars(args), "ckpt_args": ckpt_args},
        )

        if args.viewer:
            print("Viewer is open. Press ESC or close the window to exit.")
            while True:
                try:
                    env.sim_env.render(render_components="viewer")
                    time.sleep(0.01)
                except SystemExit:
                    break
        safe_close_env(env)

    print(f"Saved plot to {args.save_plot}")
    if args.show_plot:
        img = plt.imread(args.save_plot)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
