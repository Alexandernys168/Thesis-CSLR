import json
import os
import collections
from collections import Counter

# Configuration
JSON_PATH = r"a:\Thesis-CSLR\ASL\1\wlasl-complete\WLASL_v0.3.json"
OUTPUT_DIR = r"a:\Thesis-CSLR\ASL"
NUM_CLASSES = 100

def create_subset():
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        content = json.load(f)

    # Content is a list of entries, where each entry is:
    # {
    #   "gloss": "word",
    #   "instances": [ { "video_id": "...", "frame_start": 1, "frame_end": 50, "split": "train" }, ... ]
    # }

    print("Counting gloss frequencies...")
    gloss_counts = {}
    for entry in content:
        gloss = entry['gloss']
        count = len(entry['instances'])
        gloss_counts[gloss] = count

    # Select top 100 glosses
    # Note: Counter.most_common sorts by count descending
    # In case of ties, the order in the original list is preserved or arbitrary. 
    # For reproducibility, you might want to sort by gloss name too, but collections.Counter is usually stable enough for this.
    counter = Counter(gloss_counts)
    top_100 = counter.most_common(NUM_CLASSES)
    
    top_glosses = {item[0] for item in top_100}
    gloss_to_idx = {gloss: i for i, (gloss, _) in enumerate(top_100)}

    print(f"Selected top {NUM_CLASSES} glosses.")
    print(f"Example top 3: {top_100[:3]}")

    train_data = []
    val_data = []

    print("Filtering instances...")
    for entry in content:
        gloss = entry['gloss']
        if gloss in top_glosses:
            label_idx = gloss_to_idx[gloss]
            
            for inst in entry['instances']:
                # Extract necessary fields
                video_id = inst['video_id']
                frame_start = inst['frame_start']
                frame_end = inst['frame_end']
                split = inst['split']
                
                sample = {
                    'video_id': video_id,
                    'gloss': gloss,
                    'label': label_idx,
                    'frame_start': frame_start,
                    'frame_end': frame_end
                }
                
                # Check for video existence (optional but recommended during dataset creation, 
                # but for now we just rely on the 'split' field in JSON)
                # The user said: "Save train_100.json and val_100.json mapping video IDs to labels and trim indices."
                
                # 'train' split generally goes to train_data
                # 'val' AND 'test' usually go to val_data for this exercise unless strictly specified otherwise.
                # However, standard practice is splitting. Let's stick to the JSON 'split'.
                # We'll merge 'test' into 'val' or just ignore 'test' if not requested.
                # User asked for "Train/Val", usually implying use the provided splits.
                
                if split == 'train':
                    train_data.append(sample)
                elif split == 'val' or split == 'test':
                     val_data.append(sample)

    # Save to files
    train_out = os.path.join(OUTPUT_DIR, 'train_100.json')
    val_out = os.path.join(OUTPUT_DIR, 'val_100.json')
    
    print(f"Saving {len(train_data)} training samples to {train_out}")
    with open(train_out, 'w') as f:
        json.dump(train_data, f, indent=4)
        
    print(f"Saving {len(val_data)} validation samples to {val_out}")
    with open(val_out, 'w') as f:
        json.dump(val_data, f, indent=4)
        
    # Also save the class mapping
    class_map_out = os.path.join(OUTPUT_DIR, 'wlasl100_classes.json')
    with open(class_map_out, 'w') as f:
        json.dump(gloss_to_idx, f, indent=4)
    print(f"Saved class mapping to {class_map_out}")

if __name__ == "__main__":
    create_subset()
