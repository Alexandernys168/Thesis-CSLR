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

# --- DYNAMIC S3 PATHS (Adjusted to your bucket structure) ---
S3_BUCKET = "alexander-thesis-cslr"
# Base directory in your bucket
BASE_S3_PATH = "datasets/wlasl2000"

# Paths relative to BASE_S3_PATH
MASTER_JSON_KEY = f"{BASE_S3_PATH}/wlasl-complete/WLASL_v0.3.json"
S3_VIDEO_PREFIX = f"{BASE_S3_PATH}/wlasl-complete/videos/" # Folder where .mp4 or .pt live

# Output path for the new metadata
S3_METADATA_PREFIX = "metadata" 

LOCAL_OUTPUT_DIR = "./metadata_subset"
NUM_CLASSES = 1000

# Initialize S3 Session
session = boto3.Session(
    aws_access_key_id=settings.AWS_SERVER_PUBLIC_KEY,
    aws_secret_access_key=settings.AWS_SERVER_SECRET_KEY,
    region_name=settings.AWS_REGION
)
s3_client = session.client('s3')

def get_available_video_ids(bucket, prefix):
    """
    Scans S3 and returns a Set of video IDs that exist on disk.
    Essential for avoiding 'File Not Found' errors during training.
    """
    print(f"🔍 Scanning s3://{bucket}/{prefix} ...")
    available_ids = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                filename = os.path.basename(obj['Key'])
                if filename:
                    # Strip extension (e.g., '00123.mp4' -> '00123')
                    video_id = os.path.splitext(filename)[0]
                    available_ids.add(video_id)
                
    print(f"✅ Found {len(available_ids)} video files in S3.")
    return available_ids

def download_master_json():
    """Fetches the master JSON from your specific S3 path."""
    local_path = os.path.join(LOCAL_OUTPUT_DIR, "WLASL_v0.3.json")
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        os.makedirs(LOCAL_OUTPUT_DIR)
        
    print(f"⬇️ Downloading master JSON from s3://{S3_BUCKET}/{MASTER_JSON_KEY}...")
    try:
        s3_client.download_file(S3_BUCKET, MASTER_JSON_KEY, local_path)
        return local_path
    except Exception as e:
        print(f"❌ Error: Master JSON not found at the specified path: {e}")
        return None

def upload_to_s3(local_file_path):
    """Uploads the resulting 1000-class JSONs back to S3."""
    filename = os.path.basename(local_file_path)
    s3_key = f"{S3_METADATA_PREFIX}/{filename}"
    try:
        s3_client.upload_file(local_file_path, S3_BUCKET, s3_key)
        print(f"🚀 Synced: s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"❌ Upload failed for {filename}: {e}")

def create_subset():
    # 1. Sync with S3 Reality
    available_videos = get_available_video_ids(S3_BUCKET, S3_VIDEO_PREFIX)
    if not available_videos:
        print("🛑 No videos found! Please check your S3_VIDEO_PREFIX.")
        return

    # 2. Get the Master Index
    local_json_path = download_master_json()
    if not local_json_path: return

    with open(local_json_path, 'r') as f:
        content = json.load(f)

    # 3. Filter Classes (Validated against S3 availability)
    print("📊 Ranking classes by available video count...")
    gloss_counts = Counter()
    for entry in content:
        gloss = entry['gloss']
        # Only count instances that we actually HAVE in S3
        valid_count = sum(1 for inst in entry['instances'] if inst['video_id'] in available_videos)
        if valid_count > 0:
            gloss_counts[gloss] = valid_count

    top_1000 = gloss_counts.most_common(NUM_CLASSES)
    top_glosses = {item[0] for item in top_1000}
    gloss_to_idx = {gloss: i for i, (gloss, _) in enumerate(top_1000)}
    
    # 4. Filter Samples
    train_data, val_data = [], []
    skipped = 0

    print("🛠️  Generating subset JSONs...")
    for entry in content:
        gloss = entry['gloss']
        if gloss in top_glosses:
            label_idx = gloss_to_idx[gloss]
            for inst in entry['instances']:
                if inst['video_id'] in available_videos:
                    sample = {
                        'video_id': inst['video_id'],
                        'gloss': gloss,
                        'label': label_idx,
                        'frame_start': inst['frame_start'],
                        'frame_end': inst['frame_end']
                    }
                    if inst['split'] == 'train':
                        train_data.append(sample)
                    else: # val and test
                        val_data.append(sample)
                else:
                    skipped += 1

    print(f"✅ Subset Ready. (Skipped {skipped} missing videos)")
    print(f"📈 Final Counts -> Train: {len(train_data)} | Val: {len(val_data)}")

    # 5. Export and Upload
    paths = {
        'train_1000.json': train_data,
        'val_1000.json': val_data,
        'wlasl1000_classes.json': gloss_to_idx
    }

    for name, data in paths.items():
        local_p = os.path.join(LOCAL_OUTPUT_DIR, name)
        with open(local_p, 'w') as f:
            json.dump(data, f, indent=4)
        upload_to_s3(local_p)

if __name__ == "__main__":
    create_subset()