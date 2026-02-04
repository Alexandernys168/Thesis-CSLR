import json
import os
import boto3
from collections import Counter
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()

class Settings:
    AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SERVER_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

settings = Settings()

# --- DYNAMIC AWS PATHS ---
# Path where kagglehub stores the downloaded dataset
BASE_KAGGLE_DIR = "/home/sagemaker-user/.cache/kagglehub/datasets/sttaseen/wlasl2000-resized/versions/1"

JSON_PATH = os.path.join(BASE_KAGGLE_DIR, "wlasl-complete/WLASL_v0.3.json")
# Local temporary directory to store the new JSON files before upload
LOCAL_OUTPUT_DIR = "./metadata_subset" 
S3_BUCKET = "alexander-thesis-cslr"
S3_PREFIX = "metadata" # Folder in S3 for JSONs

NUM_CLASSES = 1000

# Initialize S3 Session
session = boto3.Session(
    aws_access_key_id=settings.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=settings.AWS_SERVER_SECRET_KEY,
    region_name=settings.AWS_REGION
)
s3_client = session.client('s3')

def upload_to_s3(local_file_path):
    """Uploads a file to S3 and prints status."""
    filename = os.path.basename(local_file_path)
    s3_key = f"{S3_PREFIX}/{filename}"
    try:
        s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
        print(f"✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"❌ Failed to upload {filename}: {e}")

def create_subset():
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        os.makedirs(LOCAL_OUTPUT_DIR)

    print(f"Loading {JSON_PATH}...")
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found. Check your kagglehub path.")
        return

    with open(JSON_PATH, 'r') as f:
        content = json.load(f)

    print("Counting gloss frequencies...")
    gloss_counts = {entry['gloss']: len(entry['instances']) for entry in content}

    counter = Counter(gloss_counts)
    top_1000 = counter.most_common(NUM_CLASSES)
    
    top_glosses = {item[0] for item in top_1000}
    gloss_to_idx = {gloss: i for i, (gloss, _) in enumerate(top_1000)}

    train_data = []
    val_data = []

    print("Filtering instances...")
    for entry in content:
        gloss = entry['gloss']
        if gloss in top_glosses:
            label_idx = gloss_to_idx[gloss]
            for inst in entry['instances']:
                sample = {
                    'video_id': inst['video_id'],
                    'gloss': gloss,
                    'label': label_idx,
                    'frame_start': inst['frame_start'],
                    'frame_end': inst['frame_end']
                }
                if inst['split'] == 'train':
                    train_data.append(sample)
                elif inst['split'] in ['val', 'test']:
                    val_data.append(sample)

    # Local file paths
    train_out = os.path.join(LOCAL_OUTPUT_DIR, 'train_1000.json')
    val_out = os.path.join(LOCAL_OUTPUT_DIR, 'val_1000.json')
    class_map_out = os.path.join(LOCAL_OUTPUT_DIR, 'wlasl1000_classes.json')

    # Save and Upload
    files_to_process = [
        (train_data, train_out),
        (val_data, val_out),
        (gloss_to_idx, class_map_out)
    ]

    for data, path in files_to_process:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        upload_to_s3(path)

    print("All metadata files processed and synced to S3.")

if __name__ == "__main__":
    create_subset()