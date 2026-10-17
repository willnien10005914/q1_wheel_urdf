"""Native skating uses measured contact forces and body-relative wheel motion."""
import math
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_apply
from isaaclab.utils import configclass
from ..inertia import native_skills as base
from wheel_humanoid_lab.assets.q1_inertia import Q1_INERTIA_POSITION_JOINTS
from .gait import profile, leg_angles, measured_rewards, CycleTracker, failure_mask


class SlideCommand(base.NativeSkillCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sliding = cfg.sliding
        self.wheel_sensor_ids, _ = self.sensor.find_bodies(['left_wheel_link', 'right_wheel_link'], preserve_order=True)
        self.cycles = CycleTracker(self.num_envs, self.device)
        self.push_reward = torch.zeros(self.num_envs, device=self.device)
        self.return_reward = self.push_reward.clone(); self.hop_penalty = self.push_reward.clone()
        self.normal = torch.zeros(self.num_envs, 2, device=self.device)
        self.relative_vx = self.normal.clone(); self.clearance = self.normal.clone()
        self.force_x = self.normal.clone(); self.stance = self.normal.bool()

    def set_mode(self, name, env_ids=None):
        self.sliding = name == 'slide' or (name == 'walk' and self.cfg.sliding)
        super().set_mode('walk' if name == 'slide' else name, env_ids)
        if hasattr(self, 'cycles'):
            ids = self.robot._ALL_INDICES if env_ids is None or isinstance(env_ids, slice) else env_ids
            self.cycles.reset(ids)

    def _update_command(self):
        super()._update_command()
        if self.sliding:
            angle = self.elapsed*(2*math.pi/1.4)
            self._command[:, 5] = torch.sin(angle)
            self._command[:, 6] = torch.cos(angle)
            self._command[:, 7] = self.level

    def reference(self):
        if not self.sliding: return super().reference()
        _, _, x, lift = profile(self.elapsed)
        speed = self._env.command_manager.get_command('base_velocity')[:, 0]
        blend = base.smooth(self.elapsed/1.8)*self.level*(speed.abs()/.25).clamp(0, 1)
        hip, knee = leg_angles(x*blend[:, None], lift*blend[:, None])
        ref = self.poses['stand'].expand(self.num_envs, -1).clone()
        for i, side in enumerate(('left', 'right')):
            sign = 1. if side == 'left' else -1.
            ref[:, Q1_INERTIA_POSITION_JOINTS.index(side+'_hip_Pitch_joint')] = sign*hip[:, i]
            ref[:, Q1_INERTIA_POSITION_JOINTS.index(side+'_knee_1_Pitch_joint')] = -sign*knee[:, i]
        return self.start_q+(ref-self.start_q)*base.smooth(self.elapsed/.35)[:, None]

    def update_state(self):
        if self.tick == self._env.common_step_counter: return
        super().update_state()
        if not self.sliding: return
        r = self.robot.data; q = r.root_quat_w[:, None, :].expand(-1, 2, -1)
        position = quat_apply_inverse(q, r.body_pos_w[:, self.feet]-r.root_pos_w[:, None, :])
        velocity = quat_apply_inverse(q, r.body_lin_vel_w[:, self.feet]-r.root_lin_vel_w[:, None, :])
        velocity -= torch.cross(r.root_ang_vel_b[:, None, :].expand_as(position), position, dim=-1)
        forces = self.sensor.data.net_forces_w[:, self.wheel_sensor_ids]
        heading = quat_apply(r.root_quat_w, torch.tensor([1., 0., 0.], device=self.device).expand(self.num_envs, -1))
        heading[:, 2] = 0.; heading /= heading.norm(dim=-1, keepdim=True).clamp_min(1.e-6)
        self.normal = forces[:, :, 2].clamp_min(0)
        self.force_x = (forces*heading[:, None, :]).sum(-1)
        self.relative_vx = velocity[:, :, 0]
        self.clearance = r.body_pos_w[:, self.feet, 2]-self._env.scene.env_origins[:, 2:3]-.1
        _, self.stance, _, _ = profile(self.elapsed)
        self.push_reward, self.return_reward, self.hop_penalty = measured_rewards(
            self.stance, self.normal, self.force_x, self.relative_vx, self.clearance, r.root_lin_vel_b[:, 0])
        self.cycles.update(self.stance, self.normal, self.relative_vx, self.force_x,
                           r.root_lin_vel_b[:, 0], self._env.step_dt)


@configclass
class SlideCommandCfg(base.NativeSkillCommandCfg):
    class_type: type = SlideCommand
    sliding: bool = True
    mode: str = 'walk'


class SlideAction(base.NativeSkillAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        s = self._env.command_manager.get_term('skill')
        if not s.sliding: return
        r = self.robot.data; jac = self.robot.root_physx_view.get_jacobians()
        if jac.shape[1] != self.robot.num_bodies: raise RuntimeError('Unexpected floating-base Jacobian shape')
        offset = r.body_com_pos_w-r.body_pos_w
        jcom = jac[:, :, :3, 6:]+torch.cross(jac[:, :, 3:, 6:].transpose(-2, -1), offset[:, :, None, :], dim=-1).transpose(-2, -1)
        mass = r.default_mass.to(self.device)
        gravity = 9.81*(jcom[:, :, 2, :]*mass[:, :, None]).sum(1)
        load = s.normal/(s.normal.sum(-1, keepdim=True).clamp_min(1.))
        support = (jac[:, self.feet, 2, 6:]*load[:, :, None]).sum(1)*(9.81*mass.sum(1))[:, None]
        self.feedforward.copy_((gravity-support)[:, self.pids].clamp(-45., 45.))


@configclass
class SlideActionCfg(base.NativeSkillActionCfg):
    class_type: type = SlideAction


def push(env): return base.skill(env).push_reward
def recover(env): return base.skill(env).return_reward
def cycle(env): return base.skill(env).cycles.bonus
def hop(env):
    s = base.skill(env)
    return s.hop_penalty+(s.robot.data.root_lin_vel_b[:, 2].abs()-.25).clamp_min(0)


def failure(env):
    s = base.skill(env); r = s.robot.data
    result = failure_mask(r.root_pos_w[:,2]-env.scene.env_origins[:,2], -r.projected_gravity_b[:,2],
                          s.illegal, r.root_pos_w[:,:2]-env.scene.env_origins[:,:2])
    env._native_last_illegal = s.illegal.clone()
    env._slide_last = dict(cycles=s.cycles.cycles.clone(), push=s.push_reward.clone(),
                           recovery=s.return_reward.clone(), hop=s.hop_penalty.clone(),
                           normal=s.normal.clone(), relative_vx=s.relative_vx.clone(),
                           force_x=s.force_x.clone(), stance=s.stance.clone(), clearance=s.clearance.clone())
    return result


def curriculum(env, env_ids, start_iter=0, steps_per_iteration=24):
    s = env.command_manager.get_term('skill')
    s.level = min(1., .15+(start_iter+env.common_step_counter//steps_per_iteration)/2000*.85)
    return s.level
