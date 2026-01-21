import json
import csv
import re
import os
from collections import Counter

def filter_and_intersect(how2sign_path, wlasl_path, wlasl_class_list_path, min_freq=10, max_len=17):
    # ---------------------------------------------------------
    # 1. LOAD WLASL DATA (Indices and Counts)
    # ---------------------------------------------------------
    print("Loading WLASL data...")
    
    # 1a. Load Class List (Gloss -> Index)
    wlasl_indices = {}
    try:
        with open(wlasl_class_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    idx, gloss = parts[0], parts[1]
                    wlasl_indices[gloss.upper()] = idx
        print(f"Loaded {len(wlasl_indices)} classes from {wlasl_class_list_path}")
    except FileNotFoundError:
        print(f"Error: Could not find WLASL class list at {wlasl_class_list_path}")
        return

    # 1b. Load WLASL JSON (Gloss -> Count)
    wlasl_counts = {}
    wlasl_glosses = set()
    
    try:
        if wlasl_path.endswith('.json'):
            with open(wlasl_path, 'r') as f:
                wlasl_data = json.load(f)
                
                if isinstance(wlasl_data, list):
                    for entry in wlasl_data:
                        if 'gloss' in entry:
                            g = entry['gloss'].upper()
                            count = len(entry.get('instances', []))
                            wlasl_counts[g] = count
                            wlasl_glosses.add(g)
        else:
            print(f"Error: WLASL path must be a JSON file to count instances: {wlasl_path}")
            return
            
    except FileNotFoundError:
        print(f"Error: Could not find WLASL file at {wlasl_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode WLASL JSON file at {wlasl_path}")
        return

    print(f"Loaded {len(wlasl_glosses)} glosses from WLASL JSON.")

    # ---------------------------------------------------------
    # 2. LOAD HOW2SIGN DATA (Preserving Columns)
    # ---------------------------------------------------------
    print("Loading How2Sign data...")
    
    how2sign_rows = []
    how2sign_header = []
    
    all_tokens = [] # For frequency calculation
    
    try:
        with open(how2sign_path, 'r', encoding='utf-8') as f:
            # Assuming TSV based on filename in example or CSV. 
            # The prompt implies CSV columns. Let's try CSV sniffer or assume delimiter.
            # Example in previous code used csv.reader default (comma).
            # If it fails, might need delimiter='\t'.
            
            # Let's verify delimiter quickly by peeking or just try default first.
            # Given previous code used csv.reader(f) without delimiter, we assume comma.
            
            reader = csv.reader(f)
            how2sign_header = next(reader, None)
            
            # Find the SENTENCE column.
            # Previous code found it at -1 or col 6. 
            # Let's search for header "SENTENCE"
            sentence_col_idx = -1
            if how2sign_header:
                # Try to find 'SENTENCE' exact match first
                for i, h in enumerate(how2sign_header):
                    if h.strip().upper() == 'SENTENCE':
                        sentence_col_idx = i
                        break
                
                # If not found, try partial as fallback but be careful
                if sentence_col_idx == -1:
                     for i, h in enumerate(how2sign_header):
                        if 'SENTENCE' in h.upper() and 'ID' not in h.upper() and 'NAME' not in h.upper():
                            sentence_col_idx = i
                            break

            
            if sentence_col_idx == -1:
                # Fallback to last column as per previous logic logic if not found
                sentence_col_idx = len(how2sign_header) - 1 if how2sign_header else -1
                print(f"Warning: 'SENTENCE' column not found in header. Using column index {sentence_col_idx}.")

            for row in reader:
                if len(row) > sentence_col_idx:
                    sentence = row[sentence_col_idx]
                    # Clean punctuation for tokenization
                    clean_sentence = re.sub(r'[^\w\s]', '', sentence).upper()
                    tokens = clean_sentence.split()
                    
                    how2sign_rows.append({
                        'original_row': row,
                        'tokens': tokens,
                        'clean_sentence': clean_sentence
                    })
                    all_tokens.extend(tokens)
                    
    except FileNotFoundError:
        print(f"Error: Could not find How2Sign file at {how2sign_path}")
        return

    print(f"Total How2Sign sentences loaded: {len(how2sign_rows)}")

    # ---------------------------------------------------------
    # 3. CALCULATE GLOBAL WORD FREQUENCIES (How2Sign)
    # ---------------------------------------------------------
    word_counts = Counter(all_tokens)
    print(f"Total unique glosses found in How2Sign: {len(word_counts)}")

    # ---------------------------------------------------------
    # 4. APPLY FILTERS
    # ---------------------------------------------------------
    filtered_rows = []
    target_glosses = set()
    
    dropped_len = 0
    dropped_freq = 0
    
    for item in how2sign_rows:
        tokens = item['tokens']
        
        # Criteria 1: Length check
        if len(tokens) >= max_len:
            dropped_len += 1
            continue
            
        # Criteria 2: Frequency check
        if all(word_counts[t] >= min_freq for t in tokens):
            filtered_rows.append(item['original_row'])
            target_glosses.update(tokens)
        else:
            dropped_freq += 1

    print(f"\n--- FILTER RESULTS ---")
    print(f"Sentences dropped due to length (>={max_len}): {dropped_len}")
    print(f"Sentences dropped due to frequency (<{min_freq} occurrences): {dropped_freq}")
    print(f"Sentences remaining: {len(filtered_rows)}")
    print(f"Unique glosses in filtered set: {len(target_glosses)}")

    # ---------------------------------------------------------
    # 5. INTERSECT WITH WLASL
    # ---------------------------------------------------------
    # Glosses that are in How2Sign filtered set AND in WLASL
    available_in_wlasl = target_glosses.intersection(wlasl_glosses)
    missing_from_wlasl = target_glosses - wlasl_glosses

    print(f"\n--- WLASL INTERSECTION ---")
    print(f"Glosses present in WLASL: {len(available_in_wlasl)}")
    print(f"Glosses MISSING from WLASL: {len(missing_from_wlasl)}")

    # ---------------------------------------------------------
    # 6. OUTPUT RESULTS
    # ---------------------------------------------------------
    print("\n--- SAVING RESULTS ---")
    
    if len(available_in_wlasl) > 0:
        # A. Save WLASL Subset CSV
        wlasl_output_file = 'filtered_wlasl_selected.csv'
        print(f"Saving WLASL subset to '{wlasl_output_file}'...")
        
        with open(wlasl_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: Gloss, Index (from class list), Count (from JSON)
            writer.writerow(['gloss', 'index', 'wlasl_count'])
            
            for gloss in sorted(list(available_in_wlasl)):
                # Get index, default to -1 if not in class list (though it should be if consistent)
                idx = wlasl_indices.get(gloss, "N/A")
                cnt = wlasl_counts.get(gloss, 0)
                writer.writerow([gloss, idx, cnt])

        # B. Save Filtered How2Sign CSV
        how2sign_output_file = 'filtered_how2sign.csv'
        print(f"Saving filtered How2Sign sentences to '{how2sign_output_file}'...")
        
        with open(how2sign_output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if how2sign_header:
                writer.writerow(how2sign_header)
            
            for row in filtered_rows:
                writer.writerow(row)
                
    else:
        print("Warning: No intersection found. No files saved.")

# Example Usage:
# Adjust paths as needed
if __name__ == "__main__":
    # You can update these default paths or pass them via command line args if you extend this script
    h2s_path = 'How2Sign/how2sign_train - how2sign_train.csv'
    wlasl_json_path = 'ASL/1/wlasl-complete/WLASL_v0.3.json'
    wlasl_class_path = 'ASL/1/wlasl-complete/wlasl_class_list.txt'
    
    # Check if files exist before running default
    if os.path.exists(h2s_path) and os.path.exists(wlasl_json_path) and os.path.exists(wlasl_class_path):
        filter_and_intersect(h2s_path, wlasl_json_path, wlasl_class_path)
    else:
        # For the user, I'll print a message or they can import the function
        print("Note: Default file paths in __main__ may need adjustment based on your CWD.")
