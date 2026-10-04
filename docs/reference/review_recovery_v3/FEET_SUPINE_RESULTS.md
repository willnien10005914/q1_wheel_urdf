# Supine feet-restore 6144×20k results

## Training curve
| Iter | Plant | Kneel | Stand (train metric) |
|-----:|------:|------:|---------------------:|
| 8000 | ~100% | ~99.5% | **79.2% peak** |
| 19999 | ~100% | ~99.7% | **33.9%** (kneel farm) |

## Eval (16 envs, seed 5107, stand-cut)
| Checkpoint | Stood | gofile |
|------------|------:|--------|
| `model_8000` (peak) | **12/16** | https://gofile.io/d/46SQWJMx |
| `model_19999` (final) | **0/16** (stuck stage 3 kneel) | https://gofile.io/d/clJljFUy |

Deploy default: `checkpoints/q1_recovery_v3_supine_ppo.pt` = peak8000 copy.

## Follow-up
Prone restarted with anti-kneel-farm rewards (fade kneel/lift after latch, `kneel_linger` tax, higher stand weight).
