# DreamZero Implementation And minWM Adaptation Memory

Last learned: 2026-06-09

## Source Of Truth

DreamZero lives at:

`/home/robin_wang/mondo-ai/md_dream/dreamzero`

The earlier `/data/mondo-training-dataset/dreamzero` path is only a checkpoint/cache skeleton and does not contain the implementation.

Key files:

- Docs:
  - `docs/WAN22_BACKBONE.md`
  - `docs/DATASET_TO_GEAR_AND_TRAIN.md`
  - `docs/DROID_CONVERSION.md`
  - `docs/RLINF_VENDOR.md`
- Training scripts:
  - `scripts/train/droid_training_lora.sh`
  - `scripts/train/droid_training_full_finetune_wan22_fsdp.sh`
  - `scripts/train/agibot_training.sh`
  - `scripts/train/yam_training.sh`
- Data conversion:
  - `scripts/data/convert_lerobot_to_gear.py`
- Main model path:
  - `groot/vla/model/dreamzero/base_vla.py`
  - `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
  - `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
  - `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
- Hydra configs:
  - `groot/vla/configs/model/dreamzero/vla.yaml`
  - `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf.yaml`
  - `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22.yaml`
  - `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`
  - `groot/vla/configs/data/dreamzero/droid_relative_wan22.yaml`

## What DreamZero Is

DreamZero is a Wan-based VLA/video-policy model, not just a video generator.

It wraps a Wan video DiT in `CausalWanModel` and extends the transformer sequence from video-only tokens to:

`[video_tokens | action_register | state_register]`

The model predicts two flow-matching targets:

- `video_noise_pred`: future video/dynamics noise in Wan latent space.
- `action_noise_pred`: action chunk noise decoded from the action register.

The training loss is:

`loss = dynamics_loss + action_loss`

where both are MSE against the flow-matching training target, weighted by the scheduler. `action_mask` and `has_real_action` gate the action loss.

## Backbone Variants

Wan2.1 route:

- Default model is Wan2.1-I2V-14B-480P.
- Config: `wan_flow_matching_action_tf.yaml`.
- 14B I2V style can concatenate first-frame latent to the DiT input.
- Common script settings use `image_resolution_height=176`, `image_resolution_width=320`, and `frame_seqlen=880` for the multi-view grid latent/token layout.

Wan2.2 route:

- Wan2.2-TI2V-5B support is config-only, still using `CausalWanModel`.
- Config: `wan_flow_matching_action_tf_wan22.yaml`.
- Uses `WanVideoVAE38` with 48 latent channels.
- Uses `target_video_height=160`, `target_video_width=320`.
- 160x320 gives VAE latent 10x20, patch stride `(1,2,2)`, so `frame_seqlen=50`.
- Uses CLIP first-frame context instead of first-frame latent concat (`concat_first_frame_latent=false`).

## Model Mechanics

`WANPolicyHead` owns the text encoder, image encoder, VAE, scheduler, and `CausalWanModel`.

Training path in `WANPolicyHead.forward`:

1. Batch input is already shaped by `DreamTransform`.
2. Reads `images`, `text`, `state`, `action`, `action_mask`, `has_real_action`, `embodiment_id`.
3. Converts images from `uint8` to `[-1, 1]`, resizes for Wan2.2 if configured.
4. Encodes text with Wan T5 text encoder.
5. Encodes first frame with CLIP image encoder and VAE image condition path.
6. Encodes full video with Wan VAE.
7. Samples video noise/timesteps and action noise/timesteps.
8. Adds flow-matching noise to video latents and actions.
9. Runs `CausalWanModel`.
10. Computes weighted video dynamics loss and weighted action loss.

`CausalWanModel` internals:

- Video latents are patch-embedded by `Conv3d`.
- Action tokens come from `MultiEmbodimentActionEncoder(action, timestep_action, embodiment_id)`.
- State tokens come from `CategorySpecificMLP(state, embodiment_id)`.
- Action/state tokens are concatenated after video tokens.
- RoPE is 3D for video tokens and 1D for action/state tokens.
- Blockwise causal attention aligns one image block with one action chunk and one state token.
- `action_decoder` maps the action-token slice back to continuous action noise.

Important default alignment:

- `num_frame_per_block=2`
- `num_action_per_block=24`
- `num_state_per_block=1`
- `action_horizon=24`

This means one video block maps to one 24-step action chunk plus one state token.

## Inference Mechanics

The main closed-loop path is `WANPolicyHead.lazy_joint_video_action`.

The model repeatedly does:

`observe current image block + current state -> denoise one action chunk -> execute chunk -> observe again`

Inference uses KV cache across blocks:

- `current_start_frame` tracks where the current block sits in the causal video/action sequence.
- Cache resets when language changes, when the input is a single-frame reset, or when local attention capacity is exceeded.
- At step 0, it warms the cache with the first frame.
- During denoising, it predicts both video flow and action flow; the action scheduler fully denoises the action chunk.

The user-visible policy output is:

- `action_pred`: denoised action chunk.
- `video_pred`: generated/rolled video latent output.

## Data Processing

DreamZero expects LeRobot v2 data converted into GEAR/DreamZero metadata.

The converter `scripts/data/convert_lerobot_to_gear.py` writes only `meta/`; it does not rewrite parquet or video files.

Expected dataset shape:

- `data/chunk-000/episode_000000.parquet`
- `videos/chunk-000/<camera_key>/episode_000000.mp4`
- `meta/info.json`

Generated metadata:

- `meta/modality.json`: maps state/action/video/annotation keys.
- `meta/embodiment.json`: contains the embodiment tag.
- `meta/stats.json`: feature stats for normalization.
- `meta/relative_stats_dreamzero.json`: stats for relative actions.
- `meta/tasks.jsonl`: unique language/task annotations.
- `meta/episodes.jsonl`: per-episode length/task metadata.

Dataset configs define modality keys like:

- `video.<name>`
- `state.<name>`
- `action.<name>`
- `annotation.<name>`

Those names must match `modality.json`.

`DreamTransform` builds the actual model batch:

- Multi-view videos are tiled into a single 2x2 image grid.
- DROID uses wrist view across the top row and two exterior views on the bottom row.
- State and action are concatenated and padded to `max_state_dim` and `max_action_dim`.
- Actions are normalized, masked, and must end up in `[-1, 1]`.
- Language is converted to a descriptive prompt and tokenized by UMT5.
- A negative prompt is always present for classifier-free guidance at inference.

## Relative Action Pattern

For keys in `relative_action_keys`, DreamZero converts absolute action chunks to:

`relative_action[t] = action[t] - reference_state_at_chunk_anchor`

This requires the same sub-key to exist under both state and action. Example:

- `state.joint_position`
- `action.joint_position`
- `relative_action_keys: [joint_position]`

This is important for robot adaptation because joint/pose targets transfer better when represented relative to current proprioceptive state.

## Training Entrypoints

LoRA DROID smoke/default:

`bash scripts/train/droid_training_lora.sh`

Typical overrides:

- `data=dreamzero/droid_relative`
- `model=dreamzero/vla`
- `model/dreamzero/action_head=wan_flow_matching_action_tf`
- `model/dreamzero/transform=dreamzero_cotrain`
- `train_architecture=lora`
- `num_frames=33`
- `action_horizon=24`
- `num_views=3`
- `num_frame_per_block=2`
- `num_action_per_block=24`
- `num_state_per_block=1`

Wan2.2 full finetune with FSDP:

`bash scripts/train/droid_training_full_finetune_wan22_fsdp.sh`

Typical route:

- `data=dreamzero/droid_relative_wan22`
- `model/dreamzero/action_head=wan_flow_matching_action_tf_wan22`
- `train_architecture=full`
- `fsdp="full_shard auto_wrap"`
- `image_resolution_height=160`
- `image_resolution_width=320`
- `save_lora_only=false`

AGIbot/YAM adaptation scripts:

- Load `DreamZero-AgiBot` as pretrained checkpoint.
- Set `skip_component_loading=true`.
- Set `defer_lora_injection=true` so LoRA is injected after pretrained weights load.

RLinf path:

- Vendored under `third_party/RLinf`.
- Provides a separate FSDP2 DreamZero SFT pipeline, DROID dataloader, VAE/CausalWanModel patches, and checkpoint flow.
- Local glue lives under `configs/rlinf`, `scripts/train/rlinf_*`, and `jobs/cluster_train_rlinf_*`.

## How To Adapt The Idea To minWM

minWM currently has Wan21/HY15 video world-model training, with camera-controllable paths built around camera PRoPE and `viewmats/Ks`. DreamZero suggests a separate robot-control adaptation path where robot state/action are first-class conditions.

Recommended minWM adaptation:

1. Do not use camera PRoPE if camera intrinsics/extrinsics are unavailable.
2. Add a robot dataset schema: video frames plus `robot_state`, `robot_action`, optional `contact/attached/gripper/suction` flags, and optional language.
3. Normalize state/action with per-key stats. For target-like actions, support DreamZero-style relative action stats.
4. Add state/action encoders to the DiT path:
   - MLP for state/proprio.
   - MLP or sinusoidal-time action encoder for continuous action chunks.
   - Inject as appended registers, AdaLN conditioning, or cross-attention tokens.
5. If the goal is a controllable video world model, condition video denoising on planned future action chunks and train only video loss, or video loss plus auxiliary inverse/action loss.
6. If the goal is a policy, add an action decoder/head and train DreamZero-style action flow matching.
7. Propagate action/state conditioning through all minWM phases:
   - bidirectional SFT
   - AR teacher forcing
   - ODE or CD distillation
   - DMD/self-rollout
8. During DMD/self-rollout, planned future actions or predicted policy actions must be part of the rollout state; otherwise the model degenerates toward unconditional video continuation.

For the user's 2D/robot domain:

- Base velocity and gripper state should be action/state tensors, not text-only labels.
- If suction/attachment is visually subtle, include explicit hidden-state labels such as `suction_on`, `contact`, `attached_object_id`, or gripper force/current. Otherwise the video model has weak evidence and will average or hallucinate interaction state.
- If only video is available, the result is video continuation/generation, not reliable keyboard/robot-action control.
- Text labels can help semantic task conditioning, but they cannot replace precise continuous actions for closed-loop control.

## Practical minWM Port Checklist

- Decide whether the target is `world_model(video | action,state)` or `policy(action | video,state,text)`.
- Pick backbone: Wan21 is closer to DreamZero Wan2.1; Wan2.2 support would need matching VAE/token shape support in minWM.
- Add dataset fields and collator for `state`, `action`, masks, and metadata stats.
- Implement action/state register encoder in the Wan/HY transformer wrapper.
- Add configs for `max_state_dim`, `max_action_dim`, `action_horizon`, `num_action_per_block`, `num_state_per_block`, and normalization files.
- Add loss:
  - video flow matching loss for dynamics.
  - optional action flow matching loss if training a policy head.
- Add inference API:
  - video world model: `rollout(initial_frames, action_sequence, state_sequence) -> future_frames`.
  - policy: `get_action(observation_frames, state, language) -> action_chunk`.
- Validate shapes early: video frames, latent frames, `frame_seqlen`, block count, action horizon, and state horizon must align.

