"""Independent airborne arm tuck; ground loading retains the validated no-hop profile."""
import torch
from env import Env as SingleLoadEnv, Config, DEFAULT as ORIGINAL_DEFAULT, LOW as ORIGINAL_LOW, HIGH as ORIGINAL_HIGH

# Original nineteen parameters plus airborne shoulder, elbow, and extension angle.
DEFAULT=ORIGINAL_DEFAULT+[.6,-2.4,5.8]
LOW=ORIGINAL_LOW+[-.5,-2.48,4.3]
HIGH=ORIGINAL_HIGH+[1.8,-.3,6.2]
LOW[12]=-.5
HIGH[11]=1.18
HIGH[10]=2.58
HIGH[0]=.85
HIGH[1]=2.4
LOW[7]=LOW[8]=-2.8
HIGH[7]=HIGH[8]=2.8
LOW[3]=-.5
HIGH[3]=.5

class Env(SingleLoadEnv):
    minimum_pelvis_height=.04
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self.params=torch.tensor(DEFAULT,device=self.device).expand(self.num_envs,-1).clone()

    def _commands(self,t):
        reference=super()._commands(t)
        # Initialization may call the base reference before the two extra parameters exist.
        if self.params.shape[1]<22:return reference
        p=self.params
        s=((t-self.takeoff_t-p[:,13])/.12).clamp(0,1)
        s=s*s*(3-2*s)
        extend=(self.max_progress>p[:,21])&(self.robot.data.root_lin_vel_w[:,2]<.8)
        tucked=self.has_air&(~self.landed)&(~extend)
        extending=(extend|self.landed)&self.has_air
        kneel=p[:,15]>.5
        for side in ['l','r']:
            shoulder=self.names.index(f'{side}_shoulder_pitch_joint')
            elbow=self.names.index(f'{side}_elbow_joint')
            reference[:,shoulder]=torch.where(tucked,p[:,8]+(p[:,19]-p[:,8])*s,reference[:,shoulder])
            reference[:,elbow]=torch.where(tucked,-.3+(p[:,20]+.3)*s,reference[:,elbow])
            for suffix,push_idx,tuck_idx in [('hip_pitch',5,11),('knee',6,10)]:
                i=self.names.index(f'{side}_{suffix}_joint')
                reference[:,i]=torch.where(tucked,p[:,push_idx]+(p[:,tuck_idx]-p[:,push_idx])*s,reference[:,i])
                landing=torch.where(kneel,2.24,.30) if suffix=='knee' else torch.where(kneel,-1.02,-.15)
                reference[:,i]=torch.where(extending,landing,reference[:,i])
            reference[:,shoulder]=torch.where(extending,.35,reference[:,shoulder])
            reference[:,elbow]=torch.where(extending,-1.1,reference[:,elbow])
        waist=self.names.index('waist_pitch_joint')
        reference[:,waist]=torch.where(tucked,p[:,9]+(p[:,12]-p[:,9])*s,reference[:,waist])
        reference[:,waist]=torch.where(extending,.15,reference[:,waist])
        return reference.clamp(self.limits[:,0]+.02,self.limits[:,1]-.02)

    def _get_rewards(self):
        reward=super()._get_rewards()
        r=self.robot.data
        support=torch.where(self.params[:,15]>.5,self.wcontact.any(dim=1)&self.rcontact.any(dim=1),self.wcontact.all(dim=1)).float()
        settled=torch.exp(-r.root_ang_vel_b.square().sum(dim=1)/4.-r.root_lin_vel_w[:,:2].square().sum(dim=1))
        eligible=self.landed&(self.max_progress>5.9)&(~self.failed)&(~self.prehop)
        reward+=eligible.float()*50.*(-r.projected_gravity_b[:,2]).clamp(0,1)*support*settled*self.step_dt
        return reward

    def _get_dones(self):
        self._state()
        local=self.robot.data.root_pos_w-self.scene.env_origins
        # A folded robot can legally pass below the old standing-only pelvis threshold.
        # All disallowed body contacts still terminate the episode immediately.
        failed=self.failed|self.prehop|(local[:,2]<self.minimum_pelvis_height)|(local[:,:2].norm(dim=-1)>1.8)
        return failed,self.episode_length_buf>=self.max_episode_length-1
