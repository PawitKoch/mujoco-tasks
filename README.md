# Machenta Coding Project

## Overview

Open-loop robot arm agent in MuJoCo that performs 3 distinct manipulation tasks using privileged state information.

### Task 1

Single arm picking a cube and stacking it on top of another cube on a table.

### Task 2

Dual arms handing over a "foam brick" object between one another before placing into an "enamel-coated metal bowl" on a table.

### Task 3

Single arm fully opening a a slightly open cabinet door on a table.

## Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast Python environment management and package installation.

```sh
uv sync
uv venv
uv run -m scripts.cube_stacking # e.g. cube stacking task
```
