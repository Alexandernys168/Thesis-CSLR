import kagglehub

# Download latest version
path = kagglehub.dataset_download("sttaseen/wlasl2000-resized")

print("Path to dataset files:", path)