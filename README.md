# Physics-Guided Residual Interception

This repository contains demonstration material for physics-guided residual
learning in onboard visual-servoing-based drone interception.

## Contents

- `demo/`: representative interception visualization and notes.
- `README.md`: project overview and simulator attribution.

## Demonstration

The demonstration material shows a physics-guided interception policy combining
a proportional navigation guidance prior with a learned residual correction. A
representative rollout visualization is provided in:

```text
demo/representative_interception.png
```

## Simulator Attribution

The simulation backend used in this work is based on Aerial Gym Simulator:

```text
https://github.com/ntnu-arl/aerial_gym_simulator
```

The experiments extend the simulator at the task level for visual tracking and
physics-guided interception evaluation.
