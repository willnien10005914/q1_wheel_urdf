"""Keyboard state and command routing shared by the live Q1 skill player."""
import math

class KeyState:
    DRIVE={'W','S','A','D','Q','E','UP','DOWN','LEFT','RIGHT'}
    POSES={'R':'reset','HOME':'reset','K':'kneel','U':'unbox','B':'backflip','P':'toggle_slide'}
    def __init__(self):
        self.pressed=set();self.dirty=False;self.poses=[]
    def event(self,key,pressed):
        key=key.upper()
        if pressed and key in self.POSES:
            self.poses.append(self.POSES[key]);self.pressed.clear();self.dirty=True
        elif pressed and key in {'SPACE','X','L'}:
            self.pressed.clear();self.dirty=True
        elif key in self.DRIVE:
            if pressed:self.pressed.add(key)
            else:self.pressed.discard(key)
            self.dirty=True
    def command_update(self):
        if not self.dirty and not self.pressed:return None
        self.dirty=False;k=self.pressed
        vx=1.2*bool(k&{'W','UP'})-.55*bool(k&{'S','DOWN'})
        yaw=.45*(bool(k&{'A','LEFT'})-bool(k&{'D','RIGHT'}))+.5625*(('Q' in k)-('E' in k))
        return {'vx':vx,'vy':0.,'yaw':yaw}
    def pop_pose(self,mode):
        if not self.poses:return None
        pose=self.poses.pop(0)
        return ('skate' if mode=='slide' else 'slide') if pose=='toggle_slide' else pose

class IsaacKeyboard(KeyState):
    def __init__(self):
        super().__init__()
        import carb.input
        import omni.appwindow
        self.carb=carb.input;self.interface=carb.input.acquire_input_interface()
        window=omni.appwindow.get_default_app_window()
        self.keyboard=window.get_keyboard()
        self.subscription=self.interface.subscribe_to_keyboard_events(self.keyboard,self._on_event)
    def _on_event(self,event,*args):
        if event.type in [self.carb.KeyboardEventType.KEY_PRESS,self.carb.KeyboardEventType.KEY_RELEASE]:
            self.event(event.input.name,event.type==self.carb.KeyboardEventType.KEY_PRESS)
        return True
    def close(self):
        self.interface.unsubscribe_to_keyboard_events(self.keyboard,self.subscription)
        self.pressed.clear()

def apply_drive(env,command):
    """Activate the standing skate policy for movement; leave active skills in control."""
    values={k:float(command.get(k,0.)) for k in ['vx','vy','yaw']}
    if not all(math.isfinite(v) for v in values.values()):return False
    values['vx']=max(-.55,min(1.2,values['vx']));values['yaw']=max(-.65,min(.65,values['yaw']))
    moving=any(abs(v)>1e-5 for v in values.values())
    if moving and env.mode in ['stand','unbox_hold']:
        if not env.request('skate'):return False
    if env.mode not in ['stand','skate','slide']:return False
    env.command[:,0]=values['vx'];env.command[:,1]=values['vy'];env.command[:,2]=values['yaw']
    return True
