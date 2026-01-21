import os
import csv
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.headers = [
            "Timestamp", 
            "Model_Type", 
            "Config_ID", 
            "Epoch", 
            "Train_Loss", 
            "Train_Acc", 
            "Val_Loss", 
            "Val_Acc", 
            "Best_Val_Acc",
            "Learning_Rate"
        ]
        self._initialize_log()

    def _initialize_log(self):
        """Creates the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_epoch(self, config, epoch, train_stats, val_stats, best_val_acc, current_lr):
        """
        Logs a single epoch's results.
        
        Args:
            config (dict): Configuration dictionary (for Model_Type, Config_ID).
            epoch (int): Current epoch number.
            train_stats (tuple): (train_loss, train_acc)
            val_stats (tuple): (val_loss, val_acc)
            best_val_acc (float): Current best validation accuracy.
            current_lr (float): Current learning rate.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            timestamp,
            config.get("model_type", "unknown"),
            config.get("config_id", "default"),
            epoch + 1, # 1-based logging
            f"{train_stats[0]:.4f}",
            f"{train_stats[1]:.4f}",
            f"{val_stats[0]:.4f}",
            f"{val_stats[1]:.4f}",
            f"{best_val_acc:.4f}",
            f"{current_lr:.6f}"
        ]
        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Logged epoch {epoch+1} to {self.log_file}")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def save_config_snapshot(self, config):
        """
        Saves the config dictionary as a JSON file in the checkpoint directory.
        """
        import json
        
        config_id = config.get("config_id", "default")
        checkpoint_dir = config.get("checkpoint_dir", ".")
        
        filename = f"config_snapshot_{config_id}.json"
        filepath = os.path.join(checkpoint_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"Configuration snapshot saved to {filepath}")
        except Exception as e:
            print(f"Failed to save config snapshot: {e}")

# Simple test if run directly
if __name__ == "__main__":
    logger = ExperimentLogger("test_log.csv")
    dummy_config = {"model_type": "test_net", "config_id": "debug"}
    logger.log_epoch(dummy_config, 0, (0.5, 0.8), (0.6, 0.75), 0.75)
