# force-insertion-sim

MuJoCo simulation environment for tight-clearance peg-in-hole insertion. Built on [SimCore](https://github.com/AlexanderWegenerRobotics/SimCore) for simulation and control infrastructure.

The system implements a multi-phase expert policy inspired by the wiggle-based insertion skill framework from [Wu et al., ICRA 2024](https://doi.org/10.1109/ICRA57147.2024.10610835) and collects demonstration data structured for training imitation learning policies.

---

## Overview

The robot executes a structured insertion sequence — approach, contact detection, hole search, and insertion — using Cartesian impedance control with feed-forward Lissajous force profiles to drive wiggle motion. Each episode perturbs the hole pose within configurable bounds, producing diverse demonstrations across varying alignment conditions.

The data pipeline records synchronized observations and actions at 200 Hz per episode into HDF5 files, with a dataset index for downstream consumption.

```
force-insertion-sim/
├── configs/
│   ├── global_config.yaml     ← entry point: paths + trial name
│   ├── scene_config.yaml      ← SimCore scene: robot, hole, cameras
│   ├── task_config.yaml       ← episode parameters, wiggle profile, data output
│   └── control/
│       └── panda_arm.yaml     ← controller gains
├── src/
│   ├── main.py                ← entry point
│   ├── task/
│   │   ├── insertion_task.py  ← outer loop: N episodes, pose sampling
│   │   ├── insertion_episode.py ← state machine: APPROACH → CONTACT → SEARCH → INSERT
│   │   └── trajectory.py      ← minimum-jerk Cartesian trajectory planner
│   ├── data/
│   │   └── episode_data_collector.py ← HDF5 writer + dataset index
│   └── utils/
│       └── sensor_callback.py ← FT sensor readout with gravity compensation
├── models/
│   ├── mujoco/
│   │   ├── franka_fr3/        ← FR3 arm with peg attachment
│   │   └── props/holes/       ← hole fixture meshes (see Mesh Library)
│   └── urdf/                  ← URDF for Pinocchio kinematics
├── obs/                       ← collected dataset (gitignored)
└── tests/
    ├── data_sanity_check.ipynb
    └── read_obs.ipynb
```

---

## Installation

```bash
# Install SimCore first
git clone https://github.com/yourname/SimCore.git
cd SimCore && pip install -e . && cd ..

# Install this project
git clone https://github.com/yourname/force-insertion-sim.git
cd force-insertion-sim
```

---

## Running

```bash
cd src
python main.py
```

Configure episode count, hole geometry, and output path in `configs/task_config.yaml`. Set `headless: True` in `configs/scene_config.yaml` for fast data collection without rendering.

---

## Task Design

### State Machine

Each episode runs a four-phase state machine:

```
APPROACH → CONTACT → SEARCH → INSERT → DONE
                                     ↘ FAILED (timeout at any phase)
```

**APPROACH** — Plans a two-segment minimum-jerk trajectory: transit to a hover point above the hole, then descend to the approach pose. Both the hole pose and the approach pose are independently perturbed at the start of each episode to produce diverse initial conditions. Completion is verified by checking peg tip position against the target.

**CONTACT** — Applies a constant downward feed-forward force until the F/T sensor registers sustained contact above threshold (`n_confirm` consecutive steps). Records the contact z-height `xz0` used as the search reference.

**SEARCH** — Holds a fixed Cartesian reference at the hole surface while applying a 6D Lissajous wiggle force:

```
Fff,i(t) = aᵢ · sin(2π fᵢ t + φᵢ)
```

Hole detection is triggered when the peg tip drops more than `hole_detection_threshold` below `xz0`, indicating the peg has entered the hole.

**INSERT** — Continues wiggling with a constant downward press. A moving z-score detector monitors the peg's z-position to estimate contact state (STUCK / UNSTUCK / ALIGNED), switching to pure downward push when alignment is detected. Episode succeeds when the peg tip crosses the insertion depth threshold.

### Pose Perturbation

Two independent perturbation stages generate dataset diversity:

- **Hole pose** — sampled once per episode from a Gaussian around the nominal fixture position (configurable `pos_std`, `angle_std` around z)
- **Approach pose** — independently perturbed within tighter bounds, simulating imperfect pre-insertion positioning

Both use axis-angle perturbation composed with the nominal orientation, output in wxyz quaternion convention.

---

## Data Format

Each episode is saved as an HDF5 file under `obs/episode_XXXX/episode.h5`. A `dataset_index.yaml` in `obs/` indexes all episodes with metadata for filtered loading.

### HDF5 Schema

| Group | Signal | Shape | Description |
|-------|--------|-------|-------------|
| `obs` | `f_ext` | `(T, 3)` | External force at EE [N], gravity-compensated |
| `obs` | `f_internal` | `(T, 6)` | Internal wrench from joint torques [N, Nm] |
| `obs` | `ee_velocity` | `(T, 6)` | EE linear + angular velocity [m/s, rad/s] |
| `obs` | `timestamps` | `(T,)` | Simulation time [s] |
| `action` | `Fff` | `(T, 6)` | Feed-forward force command [N, Nm] |
| `debug` | `ee_pose` | `(T, 7)` | EE pose as [pos(3), quat_wxyz(4)] |
| `debug` | `peg_tip_pos` | `(T, 3)` | Peg tip position from MuJoCo sensor [m] |
| `debug` | `mode` | `(T,)` | Expert policy phase index |
| `debug` | `q` | `(T, 7)` | Joint positions [rad] |
| `meta` | `success` | scalar | Episode outcome |
| `meta` | `duration` | scalar | Episode sim duration [s] |
| `meta` | `hole_pos` | `(3,)` | Sampled hole position [m] |
| `meta` | `hole_quat` | `(4,)` | Sampled hole orientation [wxyz] |
| `meta` | `fail_phase` | string | Phase name at failure, empty if success |
| `meta` | `n_steps` | scalar | Number of recorded timesteps |

### Loading Data

```python
from data.episode_data_collector import EpisodeDataCollector

# Load the dataset index
index = EpisodeDataCollector.load_index("obs/")
successful = [e for e in index if e["success"]]

# Load a single episode
ep = EpisodeDataCollector.load_episode(f"obs/{successful[0]['path']}")
f_ext    = ep["obs"]["f_ext"]       # (T, 3)
Fff      = ep["action"]["Fff"]      # (T, 6)
success  = ep["meta"]["success"]
```

---

## Mesh Library

8 peg geometries × 3 clearance levels = 24 peg/hole pairs. All pegs are 80 mm long; all holes are 100 mm deep.

### Pegs

| Name | Shape | Cross-section |
|------|-------|---------------|
| `peg_sq_s` | Square | 20 × 20 mm |
| `peg_sq_l` | Square | 30 × 30 mm |
| `peg_rect_s` | Rectangle | 20 × 30 mm |
| `peg_rect_l` | Rectangle | 25 × 40 mm |
| `peg_cyl_s` | Cylinder | ⌀ 20 mm |
| `peg_cyl_l` | Cylinder | ⌀ 30 mm |
| `peg_hex_s` | Hexagon | 30 mm across flats |
| `peg_hex_l` | Hexagon | 40 mm across flats |

### Holes

Each geometry is available in three clearances:

| Suffix | Clearance |
|--------|-----------|
| `_tight` | 0.2 mm/side |
| `_medium` | 0.4 mm/side |
| `_loose` | 0.7 mm/side |

Example: `hole_rect_s_tight` — rectangular hole for `peg_rect_s` with 0.2 mm clearance per side (20.4 × 30.4 mm inner).

To swap the hole geometry, update `model_path` under the `hole` object in `configs/scene_config.yaml` and adjust `hole_pose` in `configs/task_config.yaml` accordingly.

---

## Configuration

### `task_config.yaml` — key parameters

```yaml
N_task: 30                  # number of episodes per run
peg_ee_offset: 0.08         # distance from EE flange to peg tip [m]
insert_depth: 0.06          # peg tip z-drop for insertion success [m]

hole_pose:
  pos: [0.5, 0, 0.425]
  quat: [1.0, 0.0, 0.0, 0.0]
  height: 0.1               # nominal height above fixture base [m]
  pertubation:
    pos_std: [0.0035, 0.0035, 0.0]   # xy perturbation [m]
    angle_std: [0.0, 0.0, 1.0]       # rotation around z [deg]

episode:
  contact:
    force_threshold: 2.0    # contact detection threshold [N]
    f_push: 4.0             # downward push during contact phase [N]
    n_confirm: 75           # consecutive steps above threshold to confirm

  search:
    hole_detection_threshold: 0.002  # z-drop to trigger INSERT [m]
    wiggle:
      a: [7.0, 7.0, 4.931, 0.766, 0.906, 5.5]   # amplitudes [N, Nm]
      f: [1.179, 1.561, 0.0, 0.718, 0.720, 0.4]  # frequencies [Hz]
      phi: [-0.078, 0.776, 0.0, -1.562, 0.610, -0.119]  # phases [rad]
      az: 3.0               # constant downward force [N]
```

---

## Sensors

A `SensorCallback` registers with SimCore's physics loop and reads MuJoCo sensors each step. Gravity compensation rotates the gravitational load of the EE + peg into the sensor frame and subtracts it from the raw F/T reading, using the site orientation from `mj_data`.

Required MuJoCo sensors in the robot XML (prefixed by device name):

```xml
<sensor>
  <force  name="arm/ft_force"    site="arm/ft_sensor"/>
  <torque name="arm/ft_torque"   site="arm/ft_sensor"/>
  <framelinvel name="arm/ee_linvel" objtype="site" objname="arm/ee_site"/>
  <frameangvel name="arm/ee_angvel" objtype="site" objname="arm/ee_site"/>
  <framepos name="arm/peg_tip_pos" objtype="site" objname="arm/peg_tip"/>
</sensor>
```

---

## Reference

The wiggle motion and contact state estimation logic is adapted from:

> Wu et al., *1 kHz Behavior Tree for Self-adaptable Tactile Insertion*, ICRA 2024.
> DOI: [10.1109/ICRA57147.2024.10610835](https://doi.org/10.1109/ICRA57147.2024.10610835)

The simulation dataset is designed for training force-based imitation learning policies. Observation and action dimensions follow the conventions used in diffusion policy frameworks targeting contact-rich manipulation.
