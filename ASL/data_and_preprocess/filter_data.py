import json
import os

# Configuration
TRAIN_JSON = r"a:\Thesis-CSLR\ASL\train_100.json"
VAL_JSON = r"a:\Thesis-CSLR\ASL\val_100.json"
TENSOR_DIR = r"a:\Thesis-CSLR\ASL\data_tensors"

def filter_json(json_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    print(f"Filtering {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data)
    filtered_data = []
    
    for sample in data:
        video_id = sample['video_id']
        tensor_path = os.path.join(TENSOR_DIR, f"{video_id}.pt")
        
        if os.path.exists(tensor_path):
            filtered_data.append(sample)
        # else:
            # print(f"Removing missing video: {video_id}")

    new_count = len(filtered_data)
    removed_count = original_count - new_count
    
    print(f"  Original: {original_count}, Kept: {new_count}, Removed: {removed_count}")
    
    # Save back
    with open(json_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"  Updated {json_path}")

if __name__ == "__main__":
    if not os.path.exists(TENSOR_DIR):
        print("Tensor directory not found. Run preprocess.py first.")
    else:
        filter_json(TRAIN_JSON)
        filter_json(VAL_JSON)
