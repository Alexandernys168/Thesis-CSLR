# ASL Training Pipeline - Docker Instructions

This project is containerized for deployment on AWS or local Docker environments. It supports both cached tensor loading and lazy video loading.

## Prerequisites

- Docker with NVIDIA Container Toolkit support (for GPU access).
- Raw data directory on host machine (e.g., `a:\Thesis-CSLR`).

## 1. Build the Image

Run the following command from the `ASL` directory:

```bash
docker build -t asl_training_image .
```

## 2. Run the Container

You must mount your data directory to `/app/data` inside the container.

**Windows PowerShell:**
```powershell
docker run --gpus all -it --ipc=host --name asl_trainer `
  -v "a:\Thesis-CSLR:/app/data" `
  asl_training_image
```

**Linux/AWS:**
```bash
docker run --gpus all -it --ipc=host --name asl_trainer \
  -v "/path/to/host/data:/app/data" \
  asl_training_image
```

*Note: `--ipc=host` is crucial for PyTorch DataLoader shared memory.*

## 3. Configuration

Inside the container, the code expects data relative to `/app/data` if you updated `config.py` paths.

**Lazy Loading Mode:**
To save storage, set `load_mode` to "on_the_fly" in `config.py`:
```python
"load_mode": "on_the_fly",
"video_dir": r"/app/data/ASL/videos", # Ensure this matches the mounted path inside container
```

**Workers:**
When using `on_the_fly`, increase `num_workers` in `config.py` (e.g., 8 or 16) to ensure the GPU isn't waiting for video decoding.
