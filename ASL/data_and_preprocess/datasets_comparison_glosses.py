import json
from collections import Counter

def filter_and_intersect(how2sign_path, wlasl_path, min_freq=10, max_len=17):
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("Loading datasets...")
    
    # Load WLASL (Assuming standard JSON format)
    # Load WLASL (Supports JSON or TXT)
    try:
        if wlasl_path.endswith('.json'):
            with open(wlasl_path, 'r') as f:
                wlasl_data = json.load(f)
                # Check structure: List of dicts with 'gloss' key (WLASL v0.3)
                if isinstance(wlasl_data, list) and len(wlasl_data) > 0 and 'gloss' in wlasl_data[0]:
                    wlasl_glosses = set(entry['gloss'].upper() for entry in wlasl_data)
                else:
                    # Fallback: Maybe it's a simple list of strings in JSON?
                    # Or a dict? Adapt as needed.
                    print("Warning: Unknown JSON format. Assuming list of entries with 'gloss' key.")
                    wlasl_glosses = set(str(entry).upper() for entry in wlasl_data)
        else:
            # Assume Text File (One gloss per line)
            with open(wlasl_path, 'r') as f:
                wlasl_glosses = set(line.strip().upper() for line in f if line.strip())
                
    except FileNotFoundError:
        print(f"Error: Could not find WLASL file at {wlasl_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode WLASL JSON file at {wlasl_path}")
        return

    import csv
    import re
    
    # Load How2Sign (Assuming CSV with SENTENCE column at end or specific index)
    how2sign_sentences = []
    try:
        with open(how2sign_path, 'r', encoding='utf-8') as f:
            # Check if it looks like a CSV
            if how2sign_path.endswith('.csv'):
                reader = csv.reader(f)
                header = next(reader, None) # Skip header
                # Column 6 seems to be SENTENCE in the provided view (0-indexed)
                # VIDEO_ID,VIDEO_NAME,SENTENCE_ID,SENTENCE_NAME,START,END,SENTENCE
                target_col = -1 
                
                for row in reader:
                    if len(row) > 0:
                        sentence = row[-1] # Assuming last column
                        # Clean punctuation: Keep only alphanumeric and spaces
                        sentence = re.sub(r'[^\w\s]', '', sentence)
                        how2sign_sentences.append(sentence.upper())
            else:
                # Fallback to line based
                for line in f:
                    if line.strip():
                        # Clean punctuation
                        line = re.sub(r'[^\w\s]', '', line)
                        how2sign_sentences.append(line.strip().upper())
                        
    except FileNotFoundError:
        print(f"Error: Could not find How2Sign file at {how2sign_path}")
        return
    except Exception as e:
        print(f"Error reading How2Sign file: {e}")
        return

    print(f"Total How2Sign sentences loaded: {len(how2sign_sentences)}")
    if len(how2sign_sentences) > 0:
        print(f"Sample sentence: {how2sign_sentences[0]}")

    # ---------------------------------------------------------
    # 2. CALCULATE GLOBAL WORD FREQUENCIES
    # ---------------------------------------------------------
    # We need to know token frequency across the WHOLE dataset first
    all_tokens = []
    for sentence in how2sign_sentences:
        tokens = sentence.split()
        all_tokens.extend(tokens)
    
    word_counts = Counter(all_tokens)
    print(f"Total unique glosses found: {len(word_counts)}")
    print(f"Most common words: {word_counts.most_common(5)}")

    # ---------------------------------------------------------
    # 3. APPLY FILTERS (Length < 17 AND High Frequency)
    # ---------------------------------------------------------
    filtered_sentences = []
    target_glosses = set()
    
    dropped_len = 0
    dropped_freq = 0
    
    for sentence in how2sign_sentences:
        tokens = sentence.split()
        
        # Criteria 1: Length check
        if len(tokens) >= max_len:
            dropped_len += 1
            continue
            
        # Criteria 2: Frequency check
        # We only keep the sentence if ALL its words appear >= min_freq times globally.
        if all(word_counts[t] >= min_freq for t in tokens):
            filtered_sentences.append(sentence)
            target_glosses.update(tokens)
        else:
            dropped_freq += 1

    print(f"\n--- FILTER RESULTS ---")
    print(f"Sentences dropped due to length (>={max_len}): {dropped_len}")
    print(f"Sentences dropped due to frequency (<{min_freq} occurrences): {dropped_freq}")
    print(f"Sentences remaining after filtering: {len(filtered_sentences)}")
    print(f"Unique glosses in these sentences: {len(target_glosses)}")

    # ---------------------------------------------------------
    # 4. INTERSECT WITH WLASL
    # ---------------------------------------------------------
    # Which of our target glosses are actually in WLASL?
    available_in_wlasl = target_glosses.intersection(wlasl_glosses)
    missing_from_wlasl = target_glosses - wlasl_glosses

    print(f"\n--- WLASL INTERSECTION ---")
    print(f"Glosses present in WLASL: {len(available_in_wlasl)}")
    print(f"Glosses MISSING from WLASL: {len(missing_from_wlasl)}")
    
    # ---------------------------------------------------------
    # 5. OUTPUT RECOMMENDATION
    # ---------------------------------------------------------
    print("\n--- RECOMMENDATION ---")
    if len(available_in_wlasl) > 0:
        print(f"You should pre-train on the {len(available_in_wlasl)} WLASL classes found.")
        print("Saving these classes to 'target_wlasl_subset.txt'...")
        with open('target_wlasl_subset.txt', 'w') as f:
            for gloss in sorted(list(available_in_wlasl)):
                f.write(f"{gloss}\n")
                
        print("Saving filtered sentences to 'filtered_how2sign.txt'...")
        with open('filtered_how2sign.txt', 'w') as f:
            for sent in filtered_sentences:
                f.write(f"{sent}\n")
    else:
        print("Warning: No intersection found. Check your normalization (uppercase/lowercase).")

# Example Usage:
filter_and_intersect('C:/Thesis/Thesis-CSLR/How2Sign/how2sign_train - how2sign_train.csv', 'Thesis-CSLR/ASL/1/wlasl-complete/WLASL_v0.3.json')