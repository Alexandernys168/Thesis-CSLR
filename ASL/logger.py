import os
import csv
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.headers = [
            "Timestamp", 
            "Model", 
            "Config_ID", 
            "Epoch", 
            "Train_Loss", 
            "Train_Top1",
            "Train_Top5",
            "Train_Top10",
            "Val_Loss", 
            "Val_Top1",
            "Val_Top5",
            "Val_Top10",
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
            train_stats (tuple): (train_loss, train_acc1, train_acc5, train_acc10)
            val_stats (tuple): (val_loss, val_acc1, val_acc5, val_acc10)
            best_val_acc (float): Current best validation accuracy.
            current_lr (float): Current learning rate.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Unpack stats
        train_loss, train_acc1, train_top5, train_top10 = train_stats
        val_loss, val_acc1, val_top5, val_top10 = val_stats

        row = [
            timestamp,
            config.get("model_type", "unknown"),
            config.get("config_id", "default"),
            epoch + 1, # 1-based logging
                f"{train_loss:.4f}",
                f"{train_acc1:.2f}",
                f"{train_top5:.2f}",
                f"{train_top10:.2f}",
                f"{val_loss:.4f}",
                f"{val_acc1:.2f}",
                f"{val_top5:.2f}",
                f"{val_top10:.2f}",
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
