
## Requirements
- torch==1.12.1
- torchvision==0.13.1
- torchaudio==0.12.1
- pyyaml
- pandas
- matplotlib
- scipy

## Usage
```
python exec_parallel.py <command> <exp_name>
```

- `<command>` should be `train` or `transfer`. `train` can be used to train a neural network and `transfer` can be used to reproduce our experiments for transferring learning trajectories, in the specified way by `<exp_name>`.
- `<exp_name>` is one of the keys defined in the `config.yaml`.
- The command sequentially runs each experiment over the grid specified by the `hparams_grid` option in `config.yaml`. It can sweep the grid efficiently if we execute the command multiple times in parallel.

## Development Environment

This repository provides two Dev Container configurations for flexible development setup:

### GPU Configuration (Default)

**Location:** `.devcontainer/`

**Requirements:**
- NVIDIA GPU
- NVIDIA Container Toolkit
- Docker with GPU support

**Setup:**
```
Ctrl+Shift+P → Dev Containers: Reopen in Container
```

### CPU Configuration

**Location:** `devcontainer-cpu/.devcontainer/`

**Requirements:**
- Docker installed (works on any machine)

**Setup:**
```
1. Open VS Code
2. Press Ctrl+Shift+P
3. Select "Dev Containers: Open Folder in Container…"
4. Choose folder: devcontainer-cpu
```

**Note:** The container automatically mounts the project root inside `/workspace`.

