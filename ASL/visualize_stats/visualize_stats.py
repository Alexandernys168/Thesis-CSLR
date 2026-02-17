import matplotlib.pyplot as plt
import csv
import os

def visualize_multistats(log_files):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), sharex=True)
    
    # Define a set of colors to cycle through
    colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple']
    
    for i, (label, file_path) in enumerate(log_files.items()):
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        epochs = []
        val_losses = []
        bleu_scores = []
        
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                for row_idx, row in enumerate(reader):
                    if not row: continue
                    try:
                        # Skip header if it exists
                        if row_idx == 0 and not row[0].isdigit():
                            continue
                            
                        # Assuming format: Epoch, Train Loss, Val Loss, BLEU, LR
                        epoch = int(row[0])
                        val_loss = float(row[2])
                        bleu = float(row[3])
                        
                        epochs.append(epoch)
                        val_losses.append(val_loss)
                        bleu_scores.append(bleu)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
        
        if not epochs:
            print(f"No valid data found in {file_path}")
            continue

        color = colors[i % len(colors)]
        
        # Plot Val Loss on first subplot
        ax1.plot(epochs, val_losses, color=color, linestyle='-', linewidth=2, label=f'{label}')
        
        # Plot BLEU on second subplot
        ax2.plot(epochs, bleu_scores, color=color, linestyle='-', linewidth=2, label=f'{label}')

    # Configure First Subplot (Val Loss)
    ax1.set_ylabel('Val Loss', fontsize=12)
    ax1.set_title('Validation Loss Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)

    # Configure Second Subplot (BLEU Score)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('BLEU Score Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    fig.tight_layout()
    
    # Save in the directory of the script
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'combined_training_visualization.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_files = {
        "3D-CNN": os.path.join(base_dir, 'training_log_how2sign_small_3dcnn.csv'),
        "3D-CNN + Attention": os.path.join(base_dir, 'training_log_how2sign_small_3dcnn_attention.csv'),
        "2D-CNN": os.path.join(base_dir, 'training_log_how2sign_small_2dcnn.csv')
    }
    visualize_multistats(log_files)
