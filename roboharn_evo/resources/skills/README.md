# RoboHarn-Evo Skill Library

The RoboHarn-Evo Skill Library is organized by runtime responsibility.

```text
Skill Library
├── planning
├── monitoring
├── recovery
│   ├── recovery-router
│   ├── recovery-primitives
│   └── recovery-workflows
└── experience
```

## Boundaries

- Planning skills maintain task memory, subtask planning, and high-level execution boundaries.
- Monitoring skills detect execution health and semantic OOD state. They do not emit recovery tool plans.
- `ood-detection` must directly output `OOD_scenario` and must not rely on runtime regex post-processing.
- `recovery-router` maps `OOD_scenario` to a recovery workflow.
- Recovery workflows compose reusable recovery primitives.
- Recovery primitives describe one low-level recovery action each.
- Experience records store traces, case summaries, learned lessons, and retrieval artifacts outside skill prompt bodies.

## Current OOD to workflow mapping

```text
object_not_visible   -> recover-object-not-visible
motion_blocked       -> recover-motion-blocked
grasp_lost           -> recover-grasp-lost
scene_drift_detected -> recover-scene-drift
requires_replan      -> recover-requires-replan
```
