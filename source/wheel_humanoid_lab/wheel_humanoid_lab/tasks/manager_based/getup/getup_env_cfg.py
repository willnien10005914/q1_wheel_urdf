"""Backward-compatible alias: face-up get-up is now the open-box sit-up (supine → kneel).

Stand-up is ``Isaac-Q1-Posture-v0``, not this task.
"""

from wheel_humanoid_lab.tasks.manager_based.unbox.unbox_env_cfg import (  # noqa: F401
    KNEEL_POSE,
    LIE_POSE,
    TUCK_POSE,
    UNBOX_KNOTS as GETUP_KNOTS,
    UNBOX_POS_ACTION_SCALE as GETUP_POS_ACTION_SCALE,
    YOGA_POSE as PUSH_POSE,
    Q1UnboxEnvCfg as Q1GetupEnvCfg,
    Q1UnboxEnvCfg_PLAY as Q1GetupEnvCfg_PLAY,
)
