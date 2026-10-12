"""Measured-pitch hip correction after real landing; physical actuators remain unchanged."""
import torch
from compact_env import Env as CompactEnv, Config, DEFAULT, LOW, HIGH

class Env(CompactEnv):
    def _commands(self,t):
        reference=super()._commands(t)
        pitch=torch.atan2(self.robot.data.projected_gravity_b[:,0],-self.robot.data.projected_gravity_b[:,2])
        supported=self.landed&(self.params[:,15]>.5)
        # Correct the trunk lean of the fixed kneel pose using the hip motors.
        correction=torch.where(supported,.6*pitch.clamp(-.5,.5),0.)
        for side in ['l','r']:
            i=self.names.index(f'{side}_hip_pitch_joint')
            reference[:,i]+=correction
        return reference.clamp(self.limits[:,0]+.02,self.limits[:,1]-.02)
