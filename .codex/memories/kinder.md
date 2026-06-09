# KinDER / kindergarden Environment Memory For minWM

Last learned: 2026-06-09

## Source Of Truth

Main KinDER environment repo:

`/home/robin_wang/kindergarden`

Git remote:

`https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden.git`

Main package:

`/home/robin_wang/kindergarden/src/kinder`

WM-side adapter / collection repo:

`/home/robin_wang/stable-worldmodel-kinder`

Local Python env that can import and run the relevant code:

`/home/robin_wang/miniforge3/envs/swm-kinder/bin/python`

Useful runtime env vars:

```bash
export KINDERGARDEN_HOME=/home/robin_wang/kindergarden
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONDONTWRITEBYTECODE=1
```

## What KinDER Is

KinDER is a Gymnasium-compatible physical reasoning benchmark for robot learning and planning. It provides object-centric simulator environments, but the normal `kinder.make(...)` API returns fixed-size vector observations for baseline compatibility.

Core files:

- `src/kinder/__init__.py`: environment registration and `kinder.make`.
- `src/kinder/core.py`: base env classes, vector/object-centric conversion, state access.
- `src/kinder/envs/kinematic2d/base_env.py`: Kinematic2D step/reset/render logic.
- `src/kinder/envs/kinematic2d/utils.py`: 2D action space, suction/grasp helpers, motion planning helpers.
- `src/kinder/envs/kinematic2d/object_types.py`: object type features.
- `src/kinder/envs/kinematic2d/motion2d.py`: `Motion2D`.
- `src/kinder/envs/kinematic2d/clutteredstorage2d.py`: `ClutteredStorage2D`.
- `scripts/collect_demos.py`: human demo collection.
- `scripts/generate_demo_video.py`: replay pickle demo into GIF.
- `src/kinder/utils.py`: `load_demo`, `find_all_demo_files`.

## Basic Environment Usage

Direct KinDER:

```python
import kinder

kinder.register_all_environments()
env = kinder.make(
    "kinder/ClutteredStorage2D-b1-v0",
    render_mode="rgb_array",
    allow_state_access=True,
)

obs, info = env.reset(seed=0)
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
image = env.render()

# Gymnasium wraps the env; state helpers are on unwrapped.
state_vec = env.unwrapped.get_state()
env.unwrapped.set_state(state_vec)

env.close()
```

Object-centric view from vector observation:

```python
obj_state = env.observation_space.devectorize(obs)
print(obj_state.pretty_str())
recovered = env.observation_space.vectorize(obj_state)
```

State/action descriptions:

```python
print(env.action_space.low)
print(env.action_space.high)
print(env.metadata["action_space_description"])
print(env.metadata["observation_space_description"])
```

Stable-worldmodel adapter:

```python
import gymnasium as gym
import stable_worldmodel.envs  # registers swm/Kinder...

env = gym.make(
    "swm/KinderClutteredStorage2D-b1-v0",
    render_mode="rgb_array",
    max_episode_steps=400,
)
obs, info = env.reset(seed=0)
print(info.keys())
# state, proprio, goal_state, goal_proprio, goal, env_name
```

## Action Space

The Kinematic2D CRV robot action is a 5D continuous Box:

| Index | Name | Meaning | Low | High |
| --- | --- | --- | --- | --- |
| 0 | `dx` | robot base x delta, positive right | -0.05 | 0.05 |
| 1 | `dy` | robot base y delta, positive up | -0.05 | 0.05 |
| 2 | `dtheta` | robot heading delta in radians, positive ccw | -0.19634955 | 0.19634955 |
| 3 | `darm` | arm extension delta, positive out | -0.1 | 0.1 |
| 4 | `vac` | absolute vacuum state, 0 off / 1 on | 0.0 | 1.0 |

Important semantics:

- `dx/dy/dtheta/darm` are relative increments.
- `vac` is not a delta; it directly sets suction/vacuum.
- Suction attaches movable objects when the end-effector suction geometry overlaps and `vac > 0.5`.
- In `Motion2D`, arm/vacuum are not needed; stable-worldmodel's A* policy fixes `darm=low[3]` and `vac=0.0`.
- In `ClutteredStorage2D`, all 5 action dims matter because the scripted expert uses base motion, heading, arm extension, and vacuum.

## State / Observation Space

KinDER state is object-centric internally, but the default env gives a fixed vector. The vector order is documented in `env.metadata["observation_space_description"]`.

Common robot features, indices 0..8:

| Index | Object | Feature |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |

`Motion2D-p2` actual introspection:

- Env id: `kinder/Motion2D-p2-v0`
- Observation/state shape: `(59,)`
- Render shape: `(750, 750, 3)`
- Vector layout:
  - robot: 0..8
  - target_region: 9..18
  - obstacle0: 19..28
  - obstacle1: 29..38
  - obstacle2: 39..48
  - obstacle3: 49..58

`ClutteredStorage2D-b1` actual introspection:

- Env id: `kinder/ClutteredStorage2D-b1-v0`
- Observation/state shape: `(38,)`
- Render shape: `(900, 1500, 3)`
- Vector layout:
  - robot: 0..8
  - shelf: 9..27
  - block0: 28..37

Stable-worldmodel adapter adds:

- `info["state"]`: full vector state, same as observation.
- `info["proprio"]`: robot-only `(x, y, theta, arm_joint, vacuum)`, shape `(5,)`.
- `info["goal_state"]`: full vector goal state.
- `info["goal_proprio"]`: robot-only goal proprio, shape `(5,)`.
- `info["goal"]`: rendered goal image.

Actual adapter introspection:

- `swm/KinderMotion2D-p2-v0`: obs `(59,)`, action `(5,)`, state `(59,)`, proprio `(5,)`, goal image `(750, 750, 3)`.
- `swm/KinderClutteredStorage2D-b1-v0`: obs `(38,)`, action `(5,)`, state `(38,)`, proprio `(5,)`, goal image `(900, 1500, 3)`.

## Demo Collection And Reading

Collect human demos:

```bash
cd /home/robin_wang/kindergarden
python scripts/collect_demos.py kinder/Motion2D-p1-v0
python scripts/collect_demos.py kinder/ClutteredStorage2D-b1-v0 --demo-dir my_demos
```

Controls:

- Mouse virtual sticks or PS5 controller.
- `W/S`: extend/retract arm.
- `Space`: toggle vacuum.
- `R`: reset.
- `G`: save.
- `Q`: quit.

Demo pickle layout:

```text
demos/<EnvName>/<seed>/<timestamp>.p
```

Pickle fields:

- `env_id`
- `timestamp`
- `seed`
- `observations`
- `actions`
- `rewards`
- `terminated`
- `truncated`

Read a demo:

```python
from pathlib import Path
from kinder.utils import load_demo

demo = load_demo(Path("/home/robin_wang/kindergarden/demos/Motion2D-p2/38/1763423944.p"))
print(demo["env_id"], demo["seed"])
print(len(demo["observations"]), len(demo["actions"]))
print(demo["observations"][0].shape, demo["actions"][0].shape)
```

Actual sample read:

- Path: `/home/robin_wang/kindergarden/demos/Motion2D-p2/38/1763423944.p`
- `env_id`: `kinder/Motion2D-p2-v0`
- `seed`: `38`
- observations: `87`
- actions: `86`
- rewards: `86`
- obs shape: `(59,)`
- action shape: `(5,)`
- terminated: `True`
- truncated: `False`

Replay a demo:

```python
import kinder
from kinder.utils import load_demo

kinder.register_all_environments()
demo = load_demo(path)
env = kinder.make(demo["env_id"], render_mode="rgb_array")
obs, _ = env.reset(seed=demo["seed"])
for action in demo["actions"]:
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Generate GIF:

```bash
cd /home/robin_wang/kindergarden
python scripts/generate_demo_video.py demos/Motion2D-p2/38/1763423944.p --output /tmp/motion2d.gif
```

## Stable-WorldModel Kinder Adapter

The stable-worldmodel fork already has KinDER adapters and collection/eval scripts:

`/home/robin_wang/stable-worldmodel-kinder`

Adapter files:

- `stable_worldmodel/envs/kinder/_utils.py`
- `stable_worldmodel/envs/kinder/motion2d.py`
- `stable_worldmodel/envs/kinder/cluttered_storage2d.py`

Registered ids:

- `swm/KinderMotion2D-v0`
- `swm/KinderMotion2D-p0-v0` ... `swm/KinderMotion2D-p5-v0`
- `swm/KinderClutteredStorage2D-v0`
- `swm/KinderClutteredStorage2D-b1-v0`
- `swm/KinderClutteredStorage2D-b3-v0`
- `swm/KinderClutteredStorage2D-b7-v0`
- `swm/KinderClutteredStorage2D-b15-v0`

The adapter keeps raw vector observation unchanged and augments `info` with state/proprio/goal fields. This is the best local starting point for WM data collection.

## Existing WM Datasets

Existing folder-format datasets:

`/home/robin_wang/stable-worldmodel-kinder/outputs/kinder_cluttered_storage2d_b1_500eps/dataset_folder`

- episodes: `500`
- total rows: `33248`
- `action`: `(33248, 5)` float32
- `state`: `(33248, 38)` float32
- `proprio`: `(33248, 5)` float32
- `goal_state`: `(33248, 38)` float32
- `goal_proprio`: `(33248, 5)` float32
- `pixels`: 33248 JPEG files, 224x224 when collected by script

`/home/robin_wang/stable-worldmodel-kinder/outputs/kinder_motion2d_500eps/dataset_folder`

- episodes in this local folder: `415` according to `ep_len.npz`
- total rows: `39856`
- `action`: `(39856, 5)` float32
- `state`: `(39856, 59)` float32
- `proprio`: `(39856, 5)` float32
- `goal_state`: `(39856, 59)` float32
- `goal_proprio`: `(39856, 5)` float32
- `pixels`: 39856 JPEG files, 224x224 when collected by script

Folder dataset layout:

```text
dataset_folder/
  ep_len.npz
  ep_offset.npz
  action.npz
  reward.npz
  terminated.npz
  truncated.npz
  step_idx.npz
  state.npz
  proprio.npz
  goal_state.npz
  goal_proprio.npz
  pixels/
    ep_<episode>_step_<step>.jpeg
```

Transition alignment warning:

- The collection scripts store an initial dummy action filled with NaN at row 0.
- For a transition model, use row `t` observation/state and row `t+1` action to predict row `t+1` observation/state.
- Equivalently drop the first row action and align:
  - input frame/state: `pixels[:-1]`, `state[:-1]`
  - action used: `action[1:]`
  - target frame/state: `pixels[1:]`, `state[1:]`

## Data Collection Commands For WM

Collect Motion2D:

```bash
cd /home/robin_wang/stable-worldmodel-kinder
export STABLEWM_HOME="$PWD/outputs/stablewm_cache"
export MPLCONFIGDIR=/tmp/matplotlib

/home/robin_wang/miniforge3/envs/swm-kinder/bin/python \
  scripts/data/collect_kinder_motion2d.py \
  --num-episodes 500 \
  --start-seed 0 \
  --num-passages 2 \
  --policy astar \
  --max-steps 300 \
  --image-size 224 \
  --out-dir outputs/kinder_motion2d_500eps \
  --artifact-limit 20 \
  --no-gif
```

Or:

```bash
bash scripts/data/collect_kinder_motion2d_500.sh
```

Collect ClutteredStorage2D b1:

```bash
cd /home/robin_wang/stable-worldmodel-kinder
export STABLEWM_HOME="$PWD/outputs/stablewm_cache"
export MPLCONFIGDIR=/tmp/matplotlib

/home/robin_wang/miniforge3/envs/swm-kinder/bin/python \
  scripts/data/collect_kinder_cluttered_storage2d.py \
  --num-episodes 500 \
  --start-seed 0 \
  --num-blocks 1 \
  --policy scripted \
  --max-steps 400 \
  --image-size 224 \
  --out-dir outputs/kinder_cluttered_storage2d_b1_500eps \
  --artifact-limit 20 \
  --no-gif
```

Or:

```bash
bash scripts/data/collect_kinder_cluttered_storage2d_500.sh
```

## Stable-WorldModel Training Notes Are Reference Only

The user clarified that WM training should happen inside `minWM`, not inside
`stable-worldmodel-kinder`.

Use `stable-worldmodel-kinder` for:

- environment wrappers,
- scripted/demo collection,
- already-collected folder-format datasets,
- field naming reference: `pixels`, `action`, `proprio`, `state`, goal fields.

Do not treat `scripts/train/lewm.py` as the target training path unless the user
explicitly asks for the stable-worldmodel baseline.

Data configs:

- `scripts/train/config/data/kinder_motion2d.yaml`
- `scripts/train/config/data/kinder_cluttered_storage2d.yaml`

Both load:

- `pixels`
- `action`
- `proprio`
- `state`

and cache:

- `action`
- `proprio`
- `state`

LEWM training command pattern:

```bash
cd /home/robin_wang/stable-worldmodel-kinder
export STABLEWM_HOME="$PWD/outputs/stablewm_cache"
export MPLCONFIGDIR=/tmp/matplotlib

/home/robin_wang/miniforge3/envs/swm-kinder/bin/python scripts/train/lewm.py \
  data=kinder_cluttered_storage2d \
  data.dataset.name=outputs/kinder_cluttered_storage2d_b1_500eps/dataset_folder \
  output_model_name=lewm_kinder_cluttered_storage2d_b1_500eps
```

Motion2D reference command:

```bash
/home/robin_wang/miniforge3/envs/swm-kinder/bin/python scripts/train/lewm.py \
  data=kinder_motion2d \
  data.dataset.name=outputs/kinder_motion2d_500eps/dataset_folder \
  output_model_name=lewm_kinder_motion2d_500eps
```

## Current minWM Dataset / Trainer Surface

Target repo:

`/home/robin_wang/minWM`

Relevant Wan21 files:

- `Wan21/wan_utils/dataset.py`
- `Wan21/wan_train.py`
- `Wan21/wan_trainer/camera_ar_diffusion.py`
- `Wan21/wan_trainer/camera_bidirectional_diffusion.py`
- `Wan21/wan_trainer/camera_naive_cd.py`
- `Wan21/wan_trainer/camera_dmd.py`
- `Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py`

Current Wan21 data formats:

- `LatentLMDBDataset` expects LMDB keys:
  - `latents`
  - `prompts`
- `CameraLatentLMDBDataset` expects:
  - `latents`
  - `prompts`
  - `intrinsics`
  - `poses`
  and returns `clean_latent`, `viewmats`, `Ks`.
- `CameraODERegressionLMDBDataset` expects ODE latents plus already-built
  `viewmats` and `Ks`.

Current camera trainers pass `viewmats` and `Ks` through the model for PRoPE.
That is camera-control semantics, not robot-action semantics. KinDER has no
camera parameters. Do not fake robot actions as camera poses unless the goal is
only a quick engineering smoke test.

Relevant HY15 files:

- `HY15/trainer/dataset/ti2v_dataset.py`
- `HY15/trainer/dataset_camera/ar_camera_plucker_dataset.py`

HY15 `CameraPluckerDataset` also builds `viewmats/Ks`, and its `action` field is
derived by discretizing camera poses. It is not the same as KinDER's continuous
5D robot action.

## minWM Target For KinDER WM

For minWM action-conditioned video WM, the useful columns are:

- `pixels`: video frames; current stable scripts resize to 224x224.
- `action`: 5D continuous robot command.
- `proprio`: compact robot state `(x, y, theta, arm_joint, vacuum)`.
- `state`: full object-centric vector, useful for supervision/debugging.
- `goal_state/goal_proprio/goal`: optional goal conditioning.

The minWM adaptation should support two data sources.

### Source A: Direct KinDER Built-In Demo Pickles

Source path pattern:

`/home/robin_wang/kindergarden/demos/<EnvName>/<seed>/<timestamp>.p`

Use `kinder.utils.load_demo(path)`.

Built-in demo pickles store vector observations and actions, but not rendered
pixels. For video WM training, replay each demo with the same seed and render
frames:

```python
import kinder
from kinder.utils import load_demo

kinder.register_all_environments()
demo = load_demo(path)
env = kinder.make(
    demo["env_id"],
    render_mode="rgb_array",
    allow_state_access=True,
)

obs, _ = env.reset(seed=demo["seed"])
frames = [env.render()]
states = [env.unwrapped.get_state()]

for action in demo["actions"]:
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    states.append(env.unwrapped.get_state())
    if terminated or truncated:
        break
```

Alignment for built-in demos:

- `demo["observations"][i]` aligns with `frames[i]`.
- `demo["actions"][i]` maps `frames[i] -> frames[i + 1]`.
- There is no dummy NaN action row in built-in KinDER demos.

This path is best when:

- using the original human/demo distribution,
- regenerating images at custom resolution,
- deriving extra state/proprio from `env.unwrapped.get_state()`.

### Source B: Stable-WorldModel Collected Folder Datasets

Source path examples:

- `/home/robin_wang/stable-worldmodel-kinder/outputs/kinder_cluttered_storage2d_b1_500eps/dataset_folder`
- `/home/robin_wang/stable-worldmodel-kinder/outputs/kinder_motion2d_500eps/dataset_folder`

These already contain resized JPEG pixels plus action/state/proprio arrays. Use
`ep_len.npz` and `ep_offset.npz` to slice episodes.

Alignment for stable folder datasets:

- The collection scripts store an initial dummy action filled with NaN at step 0.
- For transition training, row `t + 1` action maps row `t` frame/state to row
  `t + 1` frame/state.
- Drop or mask the first dummy NaN action.

Recommended raw clip construction:

1. Iterate each episode using `ep_len` and `ep_offset`.
2. Load JPEG frames in `pixels/ep_<ep>_step_<step>.jpeg`.
3. Build transition-aligned clips:
   - frames: `pixels[t:t+T]`
   - actions: `action[t+1:t+T]`
   - proprio: `proprio[t:t+T]`
   - optional target frames: `pixels[t+1:t+T+1]`
4. Drop or mask the first dummy NaN action.
5. Normalize actions using fixed bounds:
   - low `[-0.05, -0.05, -0.19634955, -0.1, 0.0]`
   - high `[0.05, 0.05, 0.19634955, 0.1, 1.0]`
6. Decide whether to train:
   - video-only dynamics: predict next video conditioned on action/proprio.
   - video + state dynamics: also predict full vector state/proprio.
   - goal-conditioned policy/WM: include goal image/state.

## minWM Implementation Direction

Data-layer options:

1. Add a KinDER raw/demo dataset reader in minWM that emits frames, text prompt,
   actions, proprio, and state. This is useful before VAE pre-encoding and for
   debugging alignment.
2. Add a KinDER latent LMDB preprocessor, mirroring
   `Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py`, but storing
   robot fields instead of camera fields:
   - `latents`
   - `prompts`
   - `actions`
   - `proprio`
   - `states`
   - optional `goal_latents`, `goal_state`, `goal_proprio`
3. Add an `ActionLatentLMDBDataset` in `Wan21/wan_utils/dataset.py` or a nearby
   file, returning:
   - `clean_latent`
   - `prompts`
   - `actions`
   - `proprio`
   - optional `states`

Model/trainer-layer work:

- Current Wan21 camera trainers condition through `viewmats/Ks`; KinDER needs a
  separate continuous action/proprio conditioning path.
- The action tensor should preserve the 5D continuous robot command instead of
  discretizing it like camera actions.
- A minimal first trainer can mirror `camera_ar_diffusion.py`, but replace
  `CameraLatentLMDBDataset` with an action dataset and pass `actions/proprio`
  into the generator/model through a new conditioning module.
- DreamZero-style action conditioning is a useful reference if adding action
  tokens/registers to the transformer.

Temporal compression caveat:

- Wan VAE compresses time. The WorldPlayGen preprocessor maps 77 raw frames to
  20 latent frames (`(77 - 1) / 4 + 1`).
- KinDER actions are per simulator step. If encoding long raw frame clips into
  Wan latents, decide how actions map to latent frames:
  - sample actions at latent anchor frames,
  - aggregate the 4 raw actions inside each latent interval,
  - or pass a small action sequence per latent block if the model supports it.
- Do not silently drop this alignment choice; it changes the learned dynamics.

Recommended first minWM experiment:

1. Start with `ClutteredStorage2D-b1` stable folder data because it has real
   interaction and vacuum attachment.
2. Build a conversion script that creates short clips with aligned `frames`,
   `actions`, `proprio`, and `state`.
3. Encode frames through the Wan VAE into a new action-aware LMDB schema.
4. Add `ActionLatentLMDBDataset` and a small action conditioning module.
5. Use `Motion2D-p2` as the fast debugging domain; use `ClutteredStorage2D-b1`
   to verify interaction.

Important domain notes:

- `vacuum` is a hidden interaction variable with large semantic effect but small visual change; keep it as both action and proprio/state.
- `ClutteredStorage2D` is the better first domain for interaction because it has object attachment and shelf insertion.
- `Motion2D` is simpler for debugging action-conditioned robot/base motion, but arm/vacuum are mostly irrelevant.
- For minWM, use action/proprio conditioning rather than text-only labels; text can be optional task conditioning but not a substitute for the continuous action.
