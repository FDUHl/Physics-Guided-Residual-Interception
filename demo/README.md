# Demo

Use the common launcher from the repository root:

```bash
bash demo/run_demo.sh v38 viewer
```

Replace `v38` with `v47`, `v48`, or `v49`. Replace `viewer` with `headless`
to run without the Isaac Gym window.

The launcher forwards additional command-line options to the Python demo:

```bash
bash demo/run_demo.sh v48 viewer \
  --target-rear-bearing-deg -15 \
  --init-target-pitch-deg 10
```

Files in this directory:

- `demo_png_vy_vz_residual.py`: MLP policy inference for v38 and v48.
- `demo_png_vy_lstm_pd_height.py`: recurrent policy inference for v47 and v49.
- `demo_runtime.py`: inference-only network and PNG guidance components.
- `viewer_utils.py`: Isaac Gym viewer helpers.
- `run_demo.sh`: common launcher for all four policies.
- `representative_interception.png`: representative rollout visualization.
