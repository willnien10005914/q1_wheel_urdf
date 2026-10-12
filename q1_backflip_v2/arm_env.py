"""Landing balance with smooth arm recovery and an arms-down success gate."""
import torch
from landing_env import Env as LandingEnv, Config, DEFAULT, LOW, HIGH

ARM_REST = {
    'l_shoulder_pitch_joint': .10, 'r_shoulder_pitch_joint': .10,
    'l_shoulder_roll_joint': .12, 'r_shoulder_roll_joint': -.12,
    'l_shoulder_yaw_joint': 0., 'r_shoulder_yaw_joint': 0.,
    'l_elbow_joint': -.35, 'r_elbow_joint': -.35,
}

class Env(LandingEnv):
    def _arm_blend(self, t):
        x=((t-self.touchdown_t-.12)/.65).clamp(0,1)
        return x*x*x*(10-15*x+6*x*x)*self.landed.float()

    def _commands(self,t):
        ref=super()._commands(t)
        blend=self._arm_blend(t)
        for name,value in ARM_REST.items():
            i=self.names.index(name)
            ref[:,i]=ref[:,i]+blend*(value-ref[:,i])
        return ref.clamp(self.limits[:,0]+.02,self.limits[:,1]-.02)

    def _pre_physics_step(self, actions):
        previous_target=self.prev_target.clone()
        super()._pre_physics_step(actions)
        blend=self._arm_blend(self._phase())
        prior=self._commands(self._phase())
        for name in ARM_REST:
            i=self.names.index(name)
            # Flight retains the learned swing; touchdown smoothly reduces the
            # residual to 0.04 rad so the actor cannot hold the arms overhead.
            desired=prior[:,i]+(.5*(1-blend)+.02*blend)*self.actions[:,i]
            velocity=self.robot.data.joint_vel_limits[:,self.pids[i]]*.8
            controlled=previous_target[:,i]+(desired-previous_target[:,i]).clamp(-velocity*self.step_dt,velocity*self.step_dt)
            controlled=controlled.clamp(self.limits[i,0],self.limits[i,1])
            self.target[:,i]=torch.where(self.landed,controlled,self.target[:,i])
        self.prev_target.copy_(self.target)

    def arm_status(self):
        ids=[self.pids[self.names.index(name)] for name in ARM_REST]
        rest=torch.tensor(list(ARM_REST.values()),device=self.device)
        error=(self.robot.data.joint_pos[:,ids]-rest).abs().amax(-1)
        speed=self.robot.data.joint_vel[:,ids].abs().amax(-1)
        return error,speed

    def _state(self):
        if self.tick==self.common_step_counter:return
        old_best=self.best_hold.clone() if hasattr(self,'best_hold') else None
        old_success=self.success.clone()
        super()._state()
        if old_best is None:return
        error,speed=self.arm_status()
        arms_ok=(error<.18)&(speed<.8)
        self.hold=torch.where(arms_ok,self.hold,torch.zeros_like(self.hold))
        self.best_hold=torch.maximum(old_best,self.hold)
        self.success=old_success|(self.hold>=2.)

    def _get_rewards(self):
        reward=super()._get_rewards()
        error,speed=self.arm_status()
        blend=self._arm_blend(self._phase())
        reward+=blend*(35.*torch.exp(-error.square()/.04)-10.*error.square()-.2*speed.square())*self.step_dt
        return reward

    def _reset_idx(self,ids):
        extra=None
        if hasattr(self,'params'):
            selected=self.robot._ALL_INDICES if ids is None else ids
            error,speed=self.arm_status()
            extra={'arm_error':float(error[selected].mean()),'arm_speed':float(speed[selected].mean())}
        super()._reset_idx(ids)
        if extra:
            self.extras['log'].update({f'Backflip/{k}':v for k,v in extra.items()})
            self.metric_history[-1].update(extra)
