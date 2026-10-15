The manager regression tool records a fixed number of policy steps and exits. It
uses the task's native RSL-RL runner for inference and accepts both legacy and
current checkpoints through `runtime_compat.py`. It never trains or saves a model.
Observation and action dimensions come from the selected task.

Run these commands with the same Isaac Python environment as training, one
simulation process at a time:

```bash
source scripts/isaac_env.sh
"$ISAAC_PYTHON" tools/check_runtime.py --require-cuda --out outputs/validation/runtime.json
"$ISAAC_PYTHON" tools/manager_regression.py capture --task Isaac-Q1-Skate-Play-v0 --checkpoint checkpoints/q1_skate_ppo.pt --seed 4107 --num_envs 1 --steps 300 --command base_velocity=0,0,0 --out outputs/validation/merged.json
```

Capture the original GitHub revision from a separate checkout or archive with
`--project-root /path/to/original`. That selects its task definitions and assets;
the capture/compatibility tool itself can remain in this checkout. Use the same
task, checkpoint, seed, commands, step count, and environment count for both runs.

```bash
"$ISAAC_PYTHON" tools/manager_regression.py capture --project-root /tmp/q1_upstream_baseline --task Isaac-Q1-Skate-Play-v0 --checkpoint /tmp/q1_upstream_baseline/checkpoints/q1_skate_ppo.pt --seed 4107 --num_envs 1 --steps 300 --command base_velocity=0,0,0 --out outputs/validation/original.json
python3 tools/manager_regression.py compare outputs/validation/original.json outputs/validation/merged.json --out outputs/validation/merge_comparison.json
```

Comparison checks the contract and checkpoint hash, finite observations/actions,
reward, root height/position, gravity, manager commands, and termination counts.
The defaults are conservative screening thresholds: observation RMSE 0.1, action
RMSE 0.05, reward relative RMSE 0.1, height RMSE 0.02 m, position RMSE 0.1 m,
gravity RMSE 0.05, command RMSE 1e-6, and no added terminations. Every threshold
is printed in the report and can be overridden with `--max-*`. A passed short
rollout is a smoke check; it does not establish long term skill success.

For a short PPO continuation, use the training command's `--no-publish` option.
Keep the experimental checkpoint under its own log directory, then capture it
with the same settings. Add `--allow-checkpoint-change` only when intentionally
comparing that new checkpoint. Independent trajectories can drift in contact
simulation, so also compare both policies on exactly the same recorded inputs:

```bash
"$ISAAC_PYTHON" tools/manager_regression.py policy-compare outputs/validation/merged.json --baseline-checkpoint checkpoints/q1_skate_ppo.pt --candidate-checkpoint logs/rsl_rl/experiment/run/model_9.pt --out outputs/validation/ppo_policy_comparison.json
```

`policy-compare` runs on CPU and uses the repository's `FrozenActor` inference
implementation (ELU actor checkpoints), including saved observation normalization.
It reports action RMSE, relative RMSE, maximum error, and mean absolute error.
It deliberately rejects incompatible observation dimensions, such as an old
24-action policy applied to the new native inertia task. Evaluate that task with
a checkpoint trained for its own observation/action contract.

The existing web validators test the actual web controller separately:

```bash
./validate_skills.sh --num_envs 1 --steps 1250 --seed 4107 --variants boot_kneel --out outputs/validation/web_boot.json
./validate_web_backflip.sh --seeds 4107 --out outputs/validation/web_backflip.json
```

The runtime doctor imports Torch but does not import Isaac Sim or create a
simulation. It reports architecture/package information, executes small CUDA
pointwise/reduction and matrix operations, and statically audits manager task
registration declarations without executing their modules. Successful CUDA
math establishes the tested PyTorch operations on that machine; simulator and
rendering compatibility require separate actual launch checks.
