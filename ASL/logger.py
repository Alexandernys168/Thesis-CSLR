import os
import csv
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file, extended_log_file=None):
        self.log_file = log_file
        self.extended_log_file = extended_log_file
        
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
        
        self.extended_headers = [
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
            "LR"
        ]
        
        self._initialize_log()

    def _initialize_log(self):
        """Creates the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        
        if self.extended_log_file and not os.path.exists(self.extended_log_file):
             with open(self.extended_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.extended_headers)

    def log_epoch(self, config, epoch, train_stats, val_stats, best_val_acc, current_lr):
        """
        Logs a single epoch's results.
        
        Args:
            config (dict): Configuration dictionary (for Model_Type, Config_ID).
            epoch (int): Current epoch number.
            train_stats (tuple): (train_loss, train_acc, [train_top5, train_top10])
            val_stats (tuple): (val_loss, val_acc, [val_top5, val_top10])
            best_val_acc (float): Current best validation accuracy.
            current_lr (float): Current learning rate.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Unpack stats safely
        train_loss = train_stats[0]
        train_acc1 = train_stats[1]
        
        val_loss = val_stats[0]
        val_acc1 = val_stats[1]
        
        # Standard Log (Backward Compatibility)
        row = [
            timestamp,
            config.get("model_type", "unknown"),
            config.get("config_id", "default"),
            epoch + 1,
            f"{train_loss:.4f}",
            f"{train_acc1:.4f}",              
            f"{val_loss:.4f}",
            f"{val_acc1:.4f}", 
            f"{best_val_acc:.4f}", 
            f"{current_lr:.6f}"
        ]
        
        
        safe_train_acc = train_acc1 / 100.0 if train_acc1 > 1.0 else train_acc1
        safe_val_acc = val_acc1 / 100.0 if val_acc1 > 1.0 else val_acc1
        safe_best_acc = best_val_acc / 100.0 if best_val_acc > 1.0 else best_val_acc
        
        row[5] = f"{safe_train_acc:.4f}"
        row[7] = f"{safe_val_acc:.4f}"
        row[8] = f"{safe_best_acc:.4f}"

        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Logged epoch {epoch+1} to {self.log_file}")
        except Exception as e:
            print(f"Failed to write to log file: {e}")
            
        # Extended Log
        if self.extended_log_file:
            train_top5 = train_stats[2] if len(train_stats) > 2 else 0.0
            train_top10 = train_stats[3] if len(train_stats) > 3 else 0.0
            
            val_top5 = val_stats[2] if len(val_stats) > 2 else 0.0
            val_top10 = val_stats[3] if len(val_stats) > 3 else 0.0
            
            extended_row = [
                timestamp,
                config.get("model_type", "unknown"),
                config.get("config_id", "default"),
                epoch + 1,
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
                with open(self.extended_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(extended_row)
            except Exception as e:
                print(f"Failed to write to extended log file: {e}")

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
