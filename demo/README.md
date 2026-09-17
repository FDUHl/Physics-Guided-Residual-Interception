# Demo

This directory contains representative demonstration material for the
physics-guided residual interception policy.

## Files

- `representative_interception.png`: rollout visualization showing distance,
  target observation, PNG prior command, executed command, and residual policy
  output over time.

## Method Snapshot

The demonstrated controller combines a proportional navigation guidance prior
with a learned residual policy. The prior provides a geometry-based interception
command, while the residual policy compensates for visual-servoing and task-level
tracking errors observed during interception.
