# Versioned task-instruction sets

`description/task_instruction/` remains the unmodified RMBench instruction source.
Researcher-authored instruction variants live here so that they cannot silently
replace the benchmark text.

Each instruction set has a stable identifier and its own directory:

```text
task_instruction_sets/<instruction_set_id>/
  manifest.json
  <task_name>.json
```

The task JSON files retain the RMBench `seen` / `unseen` schema. The manifest
records authorship, the base instruction set, design constraints, and the hash
of every upstream task instruction that was clarified.

Evaluation must select a set explicitly with `instruction_set`. The default
`rmbench_original` continues to resolve to `description/task_instruction/`.
Custom-set results must be reported separately from vanilla RMBench results.
