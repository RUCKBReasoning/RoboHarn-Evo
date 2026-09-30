# LIBERO-PRO adapter boundary

`LiberoProAdapter` is a real, duck-typed adapter for the canonical LIBERO step
contract. Importing it does not import LIBERO, robosuite, MuJoCo, a renderer,
or task assets.

The adapter accepts the canonical four-tuple
`(observation, reward, done, info)` and the compatible five-tuple
`(observation, reward, terminated, truncated, info)`. In either form,
`env.check_success()` (or an explicitly supplied equivalent) is the only task
success authority. Reward, `done`, `info`, action advancement, and Agent state
never become success heuristics.

Observations use the additive `NeutralObservation` envelope. Camera and
proprioception values retain their benchmark keys, shapes, dtypes, ranges, and
orientation; missing fields are omitted. No image is flipped, duplicated,
renamed to a nonexistent camera, or converted into a fabricated dual-arm pose;
capability metadata records the no-transform/raw-orientation rule. Because
canonical observations can contain object-state ground truth, `raw` is an
explicit allowlist containing observed camera/proprioception fields plus only
additional audited `public_raw_keys`. Unknown and object-state fields are
excluded.

Actions are passed to exactly one `env.step(action)` call. An explicit
`LiberoProActionContract` may validate type, shape, dtype, and bounds, but the
adapter never clips, normalizes, retries, chunks, or changes frames. Controller
mode, delta/absolute semantics, rotation representation, reference frame, and
gripper convention remain explicit optional facts rather than guessed values.

The RoboHarn-Evo Agent's compatibility interface is RMBench-shaped.
`roboharn_capability_report()` and
`require_roboharn_agent_compatibility()` check the actual observed fields plus
explicit semantic role mappings. They fail with the exact missing capabilities
instead of filling unavailable head/left/right cameras, dual-arm end poses, or
14-dimensional qpos actions.

`LiberoProAdapterBoundary` remains as an abstract compatibility import. Task
discovery and environment construction belong to the repository-level
benchmark integration, not this package.
