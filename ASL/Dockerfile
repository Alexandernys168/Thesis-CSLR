# Use an official PyTorch runtime as a parent image
# Check available tags on Docker Hub for specific versions. 
# This one specifies CUDA 12.1 and cuDNN 8 runtime.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

# Install system dependencies for OpenCV
# libgl1-mesa-glx and libglib2.0-0 are often required for cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Define environment variable
# This can be used in python scripts to detect if running in docker
ENV AM_I_IN_A_DOCKER_CONTAINER=true
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for visualization, otherwise optional)
# EXPOSE 80

# Run train_modular.py when the container launches
# User should override command arguments as needed
CMD ["python", "train_modular.py"]

# -----------------------------------------------------------------------------
# MOUNTING INSTRUCTIONS:
#
# You MUST mount your local data directory to the container to access data.
# The code expects data at specific locations (configurable in config.py).
#
# Recommended Run Command:
# docker run --gpus all -it --ipc=host --name asl_trainer \
#   -v "a:/Thesis-CSLR:/app/data" \
#   asl_training_image
#
# Inside config.py, ensure paths point to /app/data/...
# -----------------------------------------------------------------------------
