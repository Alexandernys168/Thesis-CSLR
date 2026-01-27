import json

JSON_PATH = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\WLASL_v0.3.json"
TARGET_ID = "70212"

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

for entry in data:
    for inst in entry['instances']:
        if inst['video_id'] == TARGET_ID:
            print(f"Found {TARGET_ID}:")
            print(f"Frame Start: {inst['frame_start']}")
            print(f"Frame End: {inst['frame_end']}")
            print(f"Split: {inst['split']}")
            break
