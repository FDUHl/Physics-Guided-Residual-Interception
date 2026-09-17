"""Inference-only policy and guidance components used by the public demos."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

LOS_WIDTH_SCALE_PX = 50.0


def ssa(angle):
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_rotate(quaternion, vector):
    """Rotate vectors with Isaac Gym xyzw quaternions."""
    q_xyz = quaternion[:, :3]
    q_w = quaternion[:, 3:4]
    intermediate = 2.0 * torch.cross(q_xyz, vector, dim=1)
    return vector + q_w * intermediate + torch.cross(
        q_xyz, intermediate, dim=1
    )


@dataclass
class CameraGimbalModel:
    image_width: int
    image_height: int
    fov_horizontal: float
    fov_vertical: float

    def pixels_to_angles(self, pixel_x, pixel_y):
        fx = self.image_width / (
            2.0 * torch.tan(torch.tensor(self.fov_horizontal) / 2.0)
        )
        fy = self.image_height / (
            2.0 * torch.tan(torch.tensor(self.fov_vertical) / 2.0)
        )
        fx = torch.as_tensor(fx, device=pixel_x.device, dtype=pixel_x.dtype)
        fy = torch.as_tensor(fy, device=pixel_y.device, dtype=pixel_y.dtype)
        return torch.nan_to_num(
            torch.stack(
                [torch.atan2(pixel_y, fy), torch.atan2(pixel_x, fx)], dim=1
            ),
            0.0,
        )


def build_env(
    num_envs,
    device,
    use_warp,
    robot_name,
    controller_name,
    robot_max_velocity,
    target_speed,
    headless=True,
):
    from aerial_gym.config.task_config.tracking_task_config import task_config
    from aerial_gym.task.tracking_task.tracking_task import TrackingTask

    task_config.num_envs = num_envs
    task_config.headless = headless
    task_config.use_warp = use_warp
    task_config.device = device
    task_config.robot_name = robot_name
    task_config.controller_name = controller_name
    task_config.max_velocity = robot_max_velocity
    task_config.target_velocity = target_speed
    return TrackingTask(task_config)


def safe_close_env(env):
    if env is None:
        return
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        try:
            close_fn()
            return
        except AttributeError:
            pass
    sim_env = getattr(env, "sim_env", None)
    close_fn = getattr(sim_env, "close", None)
    if callable(close_fn):
        close_fn()


def get_robot_yaw(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.task_config.num_envs, device=env.device)
    return env.obs_dict["robot_euler_angles"][env_ids, 2]


def build_full_action_bounds(
    device, max_velocity, max_vy, max_vz, max_yaw_rate
):
    bounds = torch.tensor(
        [max_velocity, max_velocity, 1.5 * max_velocity, max_yaw_rate],
        device=device,
        dtype=torch.float32,
    )
    bounds[1] = min(float(bounds[1]), float(max_vy))
    bounds[2] = min(float(bounds[2]), float(max_vz))
    bounds[3] = min(float(bounds[3]), float(max_yaw_rate))
    return bounds


def fixed_frame_los_obs(env, width_scale_px=LOS_WIDTH_SCALE_PX):
    rel_world = env.target_position - env.obs_dict["robot_position"]
    dx, dy, dz = rel_world.unbind(dim=1)
    horizontal = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6)
    distance = torch.linalg.vector_norm(rel_world, dim=1).clamp_min(1e-6)
    world_yaw = torch.atan2(dy, dx)
    robot_yaw = env.obs_dict["robot_euler_angles"][:, 2]
    yaw = ssa(world_yaw - robot_yaw)
    pitch = torch.atan2(dz, horizontal)
    width = width_scale_px / distance.clamp_min(0.1)
    return torch.nan_to_num(torch.stack([pitch, yaw, width], dim=1), 0.0)


def legacy_camera_los_obs(env, raw_obs):
    camera = CameraGimbalModel(
        env.task_config.image_width,
        env.task_config.image_height,
        float(env.task_config.camera_fov_horizontal),
        float(env.task_config.camera_fov_vertical),
    )
    raw_obs = torch.nan_to_num(raw_obs, 0.0)
    return torch.cat(
        [camera.pixels_to_angles(raw_obs[:, 0], raw_obs[:, 1]), raw_obs[:, 2:3]],
        dim=1,
    )


def raw_obs_to_policy_obs(env, raw_obs, obs_mode="fixed_frame_los"):
    if obs_mode == "legacy_camera_los":
        return legacy_camera_los_obs(env, raw_obs)
    return fixed_frame_los_obs(env)


def compute_los_rates(obs, prev_obs, dt):
    if prev_obs is None:
        pitch_dot = torch.zeros_like(obs[:, 0])
        yaw_dot = torch.zeros_like(obs[:, 1])
    else:
        pitch_dot = ssa(obs[:, 0] - prev_obs[:, 0]) / max(dt, 1e-6)
        yaw_dot = ssa(obs[:, 1] - prev_obs[:, 1]) / max(dt, 1e-6)
        pitch_dot = torch.clamp(pitch_dot, -2.5, 2.5)
        yaw_dot = torch.clamp(yaw_dot, -2.5, 2.5)
    return pitch_dot, yaw_dot


def build_policy_obs(obs, prev_obs, dt):
    pitch_dot, yaw_dot = compute_los_rates(obs, prev_obs, dt)
    return torch.cat(
        [obs[:, :3], pitch_dot[:, None], yaw_dot[:, None]], dim=1
    )


def build_yaw_rate_cmd(
    obs, prev_obs, dt, k_yaw, k_yaw_d, max_yaw_rate
):
    _, yaw_dot = compute_los_rates(obs, prev_obs, dt)
    return torch.clamp(
        k_yaw * obs[:, 1] + k_yaw_d * yaw_dot,
        -max_yaw_rate,
        max_yaw_rate,
    )


def setup_training_scene(
    env,
    env_ids=None,
    spawn_distance_min=8.0,
    spawn_distance_max=20.0,
    spawn_bearing_min_deg=-20.0,
    spawn_bearing_max_deg=20.0,
    spawn_dz_min=-1.0,
    spawn_dz_max=1.0,
    target_heading_min_deg=-180.0,
    target_heading_max_deg=180.0,
    target_min_z=1.0,
    target_max_z=3.0,
):
    if env_ids is None:
        env_ids = torch.arange(env.task_config.num_envs, device=env.device)
    if env_ids.numel() == 0:
        return

    robot_pos = env.obs_dict["robot_position"][env_ids]
    robot_quat = env.obs_dict["robot_orientation"][env_ids]
    dtype = robot_pos.dtype
    count = env_ids.numel()
    distance = torch.empty(
        count, device=env.device, dtype=dtype
    ).uniform_(spawn_distance_min, spawn_distance_max)
    bearing = torch.deg2rad(
        torch.empty(count, device=env.device, dtype=dtype).uniform_(
            spawn_bearing_min_deg, spawn_bearing_max_deg
        )
    )
    dz = torch.empty(count, device=env.device, dtype=dtype).uniform_(
        spawn_dz_min, spawn_dz_max
    )
    rel_body = torch.stack(
        [distance * torch.cos(bearing), distance * torch.sin(bearing), dz],
        dim=1,
    )
    target_pos = robot_pos + quat_rotate(robot_quat, rel_body)
    target_pos[:, 2] = torch.clamp(
        target_pos[:, 2], min=target_min_z, max=target_max_z
    )
    env.target_position[env_ids] = target_pos

    heading = torch.deg2rad(
        torch.empty(count, device=env.device, dtype=dtype).uniform_(
            target_heading_min_deg, target_heading_max_deg
        )
    )
    direction = torch.stack(
        [torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)],
        dim=1,
    )
    env.target_direction[env_ids] = direction
    target_root = env.asset_state_tensor[env_ids, env.target_asset_idx]
    target_root[:, 0:3] = target_pos
    target_root[:, 3:7] = 0.0
    target_root[:, 6] = 1.0
    target_root[:, 7:10] = direction * env.target_velocity_tensor
    target_root[:, 10:13] = 0.0


class PrivilegedPNGVyVzController:
    def __init__(
        self,
        max_vy,
        max_vz,
        width_scale_px=50.0,
        n_nav_y=2.5,
        n_nav_z=2.0,
        k_pos_y=0.9,
        k_pos_z=1.1,
        lat_range_cap=12.0,
        vert_range_cap=10.0,
        min_closing_speed=1.0,
        teacher_pos_gain=1.4,
        teacher_vel_gain=0.35,
        teacher_tgo_floor=0.35,
        pure_los_rate_prior=False,
        rate_prior_closing_speed=8.0,
        gate_vertical_rate_against_pitch=False,
    ):
        del width_scale_px, teacher_pos_gain, teacher_vel_gain, teacher_tgo_floor
        self.max_vy = float(max_vy)
        self.max_vz = float(max_vz)
        self.n_nav_y = float(n_nav_y)
        self.n_nav_z = float(n_nav_z)
        self.k_pos_y = float(k_pos_y)
        self.k_pos_z = float(k_pos_z)
        self.lat_range_cap = float(lat_range_cap)
        self.vert_range_cap = float(vert_range_cap)
        self.min_closing_speed = float(min_closing_speed)
        self.pure_los_rate_prior = bool(pure_los_rate_prior)
        self.rate_prior_closing_speed = float(rate_prior_closing_speed)
        self.gate_vertical_rate_against_pitch = bool(
            gate_vertical_rate_against_pitch
        )

    def _vertical_prior(self, pitch, pitch_dot, closing_speed):
        rate_vz = self.n_nav_z * closing_speed * pitch_dot
        if self.gate_vertical_rate_against_pitch:
            rate_vz = torch.where(
                rate_vz * pitch >= 0.0, rate_vz, torch.zeros_like(rate_vz)
            )
        return self.k_pos_z * pitch + rate_vz

    def compute_terms(self, env):
        robot_pos = env.obs_dict["robot_position"]
        robot_vel = env.obs_dict.get("robot_linvel")
        if robot_vel is None:
            robot_vel = torch.zeros_like(robot_pos)
        rel_world = env.target_position - robot_pos
        target_vel = env.target_direction * env.target_velocity_tensor
        rel_vel_world = target_vel - robot_vel

        yaw = env.obs_dict["robot_euler_angles"][:, 2]
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        x_b = cos_yaw * rel_world[:, 0] + sin_yaw * rel_world[:, 1]
        y_b = -sin_yaw * rel_world[:, 0] + cos_yaw * rel_world[:, 1]
        z_b = rel_world[:, 2]
        vx_b = cos_yaw * rel_vel_world[:, 0] + sin_yaw * rel_vel_world[:, 1]
        vy_b = -sin_yaw * rel_vel_world[:, 0] + cos_yaw * rel_vel_world[:, 1]
        vz_b = rel_vel_world[:, 2]

        horizontal = torch.sqrt(x_b * x_b + y_b * y_b).clamp_min(1e-6)
        xy_sq = (x_b * x_b + y_b * y_b).clamp_min(1e-6)
        vertical_sq = (horizontal * horizontal + z_b * z_b).clamp_min(1e-6)
        distance = torch.linalg.vector_norm(rel_world, dim=1).clamp_min(1e-6)
        closing = -torch.sum(rel_world * rel_vel_world, dim=1) / distance
        closing = torch.clamp(closing, min=self.min_closing_speed)
        los_yaw = torch.atan2(y_b, x_b)
        pitch = torch.atan2(z_b, horizontal)
        yaw_dot = (x_b * vy_b - y_b * vx_b) / xy_sq
        horizontal_dot = (x_b * vx_b + y_b * vy_b) / horizontal
        pitch_dot = (horizontal * vz_b - z_b * horizontal_dot) / vertical_sq
        return {
            "closing_speed": closing,
            "yaw": los_yaw,
            "pitch": pitch,
            "yaw_dot_true": yaw_dot,
            "pitch_dot_true": pitch_dot,
            "lat_range": torch.clamp(horizontal, max=self.lat_range_cap),
            "vert_range": torch.clamp(
                torch.sqrt(horizontal * horizontal + z_b * z_b),
                max=self.vert_range_cap,
            ),
        }

    def prior_command(self, env, policy_obs=None):
        terms = self.compute_terms(env)
        if (
            self.pure_los_rate_prior
            and policy_obs is not None
            and policy_obs.shape[1] >= 5
        ):
            pitch, los_yaw = policy_obs[:, 0], policy_obs[:, 1]
            pitch_dot, yaw_dot = policy_obs[:, 3], policy_obs[:, 4]
            closing = torch.full_like(yaw_dot, self.rate_prior_closing_speed)
            vy = self.k_pos_y * los_yaw + self.n_nav_y * closing * yaw_dot
            vz = self._vertical_prior(pitch, pitch_dot, closing)
        elif self.pure_los_rate_prior:
            closing = torch.full_like(
                terms["yaw_dot_true"], self.rate_prior_closing_speed
            )
            vy = (
                self.k_pos_y * terms["yaw"]
                + self.n_nav_y * closing * terms["yaw_dot_true"]
            )
            vz = self._vertical_prior(
                terms["pitch"], terms["pitch_dot_true"], closing
            )
        else:
            vy = (
                self.k_pos_y * terms["lat_range"] * terms["yaw"]
                + self.n_nav_y
                * terms["closing_speed"]
                * terms["yaw_dot_true"]
            )
            vz = self._vertical_prior(
                terms["vert_range"] * terms["pitch"],
                terms["pitch_dot_true"],
                terms["closing_speed"],
            )
        command = torch.stack(
            [
                torch.clamp(vy, -self.max_vy, self.max_vy),
                torch.clamp(vz, -self.max_vz, self.max_vz),
            ],
            dim=1,
        )
        return command, terms


class ResidualVyVzPolicyNet(nn.Module):
    def __init__(self, obs_dim=5, act_dim=2, hidden=(256, 256)):
        super().__init__()
        layers = []
        previous = obs_dim
        for width in hidden:
            layers.extend(
                [nn.Linear(previous, width), nn.LayerNorm(width), nn.ReLU()]
            )
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.pi = nn.Sequential(
            nn.Linear(previous, 128), nn.ReLU(), nn.Linear(128, act_dim)
        )
        self.v = nn.Sequential(
            nn.Linear(previous, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))
        self.residual_bounds = None
        self.obs_dim = obs_dim
        self.tanh_eps = 1e-6

    def forward(self, obs):
        feature = self.backbone(obs)
        return self.pi(feature), torch.exp(self.log_std), self.v(feature)

    def act(self, obs):
        mean, std, value = self(obs)
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        residual = squashed * self.residual_bounds
        log_prob = (
            distribution.log_prob(pre_tanh)
            - torch.log(1.0 - squashed.pow(2) + self.tanh_eps)
            - torch.log(self.residual_bounds)
        ).sum(-1, keepdim=True)
        mean_residual = torch.tanh(mean) * self.residual_bounds
        return residual, log_prob, value, mean_residual


class LSTMResidualVyPolicy(nn.Module):
    def __init__(self, obs_dim=5, hidden_size=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1)
        self.pi = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((1,), -1.0))
        self.residual_bounds = None
        self.tanh_eps = 1e-6

    def init_hidden(self, batch_size, device):
        zeros = torch.zeros(
            1, batch_size, self.hidden_size, device=device
        )
        return zeros.clone(), zeros.clone()

    def forward_step(self, obs, hidden):
        output, hidden = self.rnn(self.encoder(obs).unsqueeze(0), hidden)
        feature = output.squeeze(0)
        return self.pi(feature), self.v(feature), hidden

    def sample_step(self, obs, hidden):
        mean, value, hidden = self.forward_step(obs, hidden)
        distribution = Normal(mean, torch.exp(self.log_std).expand_as(mean))
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        residual = squashed * self.residual_bounds
        log_prob = (
            distribution.log_prob(pre_tanh)
            - torch.log(1.0 - squashed.pow(2) + self.tanh_eps)
        ).sum(-1)
        return residual, log_prob, value.squeeze(-1), hidden
