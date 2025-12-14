# Machenta Coding Project

## Overview

Open-loop robot arm agent in MuJoCo that performs 3 distinct manipulation tasks using privileged state information.

| Task 1 | Task 2 | Task 3 |
|:---:|:---:|:---:|
| Single arm picking a cube and stacking it on top of another cube. | Dual arms performing handover of "foam brick" into an "enamel-coated metal bowl". | Single arm opening a slightly open cabinet door. |
| ![Task 1](assets/task1.png) | ![Task 2](assets/task2.png) | ![Task 3](assets/task3.png) |

## Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast Python environment management and package installation.

```sh
uv sync
uv venv
uv run -m scripts.cube_stacking # e.g. cube stacking task
```

## Developer Notes

To generate Python type stubs for MuJoCo (for autocompletion and type checking), run:

```sh
uv run pybind11-stubgen mujoco -o typings
```

This will create or update the typings/mujoco directory with .pyi stub files for the MuJoCo Python bindings.
