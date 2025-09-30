#!/usr/bin/env python3
"""Real-time training monitoring dashboard."""

import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np
from collections import defaultdict
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, history_file="outputs/training_history.json", update_interval=5):
        self.history_file = Path(history_file)
        self.update_interval = update_interval
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16)
        
        # Initialize subplots
        self.ax_loss = self.axes[0, 0]
        self.ax_metrics = self.axes[0, 1]
        self.ax_lr = self.axes[1, 0]
        self.ax_speed = self.axes[1, 1]
        
        # Data storage
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.val_iou = []
        self.val_precision = []
        self.val_recall = []
        self.val_small_mae = []
        self.val_large_mae = []
        self.learning_rates = []
        self.speeds = []
        
        # Last update time
        self.last_update = 0
        
    def load_data(self):
        """Load training data from JSON file."""
        if not self.history_file.exists():
            return
            
        try:
            with self.history_file.open('r') as f:
                data = json.load(f)
            
            # Clear previous data
            self.epochs.clear()
            self.train_loss.clear()
            self.val_loss.clear()
            self.val_iou.clear()
            self.val_precision.clear()
            self.val_recall.clear()
            self.val_small_mae.clear()
            self.val_large_mae.clear()
            self.learning_rates.clear()
            self.speeds.clear()
            
            # Process training data
            for i, epoch_data in enumerate(data.get('train', [])):
                self.epochs.append(i + 1)
                self.train_loss.append(epoch_data.get('loss', 0))
                self.learning_rates.append(epoch_data.get('lr', 0))
                self.speeds.append(epoch_data.get('imgs_per_s', 0))
            
            # Process validation data
            for epoch_data in data.get('val', []):
                self.val_loss.append(epoch_data.get('loss', 0))
                self.val_iou.append(epoch_data.get('iou', 0))
                self.val_precision.append(epoch_data.get('precision', 0))
                self.val_recall.append(epoch_data.get('recall', 0))
                self.val_small_mae.append(epoch_data.get('small_mae', 0))
                self.val_large_mae.append(epoch_data.get('large_mae', 0))
                
        except Exception as e:
            logger.warning(f"Error loading data: {e}")
    
    def update_plots(self):
        """Update all plots with current data."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
            
        self.load_data()
        if not self.epochs:
            return
            
        self.last_update = current_time
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Loss curves
        self.ax_loss.plot(self.epochs, self.train_loss, 'b-', label='Train Loss', linewidth=2)
        if self.val_loss:
            val_epochs = self.epochs[:len(self.val_loss)]
            self.ax_loss.plot(val_epochs, self.val_loss, 'r-', label='Val Loss', linewidth=2)
        self.ax_loss.set_title('Training & Validation Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # Plot 2: Validation metrics
        if self.val_iou:
            val_epochs = self.epochs[:len(self.val_iou)]
            self.ax_metrics.plot(val_epochs, self.val_iou, 'g-', label='IoU', linewidth=2)
            self.ax_metrics.plot(val_epochs, self.val_precision, 'b-', label='Precision', linewidth=2)
            self.ax_metrics.plot(val_epochs, self.val_recall, 'r-', label='Recall', linewidth=2)
            self.ax_metrics.set_title('Validation Metrics')
            self.ax_metrics.set_xlabel('Epoch')
            self.ax_metrics.set_ylabel('Score')
            self.ax_metrics.legend()
            self.ax_metrics.grid(True, alpha=0.3)
            self.ax_metrics.set_ylim(0, 1)
        
        # Plot 3: Learning rate
        if self.learning_rates:
            self.ax_lr.plot(self.epochs, self.learning_rates, 'purple', linewidth=2)
            self.ax_lr.set_title('Learning Rate Schedule')
            self.ax_lr.set_xlabel('Epoch')
            self.ax_lr.set_ylabel('Learning Rate')
            self.ax_lr.grid(True, alpha=0.3)
            self.ax_lr.set_yscale('log')
        
        # Plot 4: Training speed and counting errors
        if self.speeds:
            self.ax_speed.plot(self.epochs, self.speeds, 'orange', label='Images/sec', linewidth=2)
            self.ax_speed.set_title('Training Speed & Counting Errors')
            self.ax_speed.set_xlabel('Epoch')
            self.ax_speed.set_ylabel('Images/sec')
            self.ax_speed.legend()
            self.ax_speed.grid(True, alpha=0.3)
            
            # Add MAE on secondary y-axis
            if self.val_small_mae and self.val_large_mae:
                ax2 = self.ax_speed.twinx()
                val_epochs = self.epochs[:len(self.val_small_mae)]
                ax2.plot(val_epochs, self.val_small_mae, 'g--', label='Small Vehicle MAE', alpha=0.7)
                ax2.plot(val_epochs, self.val_large_mae, 'r--', label='Large Vehicle MAE', alpha=0.7)
                ax2.set_ylabel('Mean Absolute Error')
                ax2.legend(loc='upper right')
        
        # Add current status text
        if self.epochs:
            current_epoch = self.epochs[-1]
            current_loss = self.train_loss[-1] if self.train_loss else 0
            current_lr = self.learning_rates[-1] if self.learning_rates else 0
            current_speed = self.speeds[-1] if self.speeds else 0
            
            status_text = f"Epoch: {current_epoch} | Loss: {current_loss:.3f} | LR: {current_lr:.2e} | Speed: {current_speed:.1f} img/s"
            self.fig.text(0.5, 0.02, status_text, ha='center', fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def animate(self, frame):
        """Animation function for matplotlib."""
        self.update_plots()
    
    def start_monitoring(self):
        """Start the real-time monitoring."""
        logger.info(f"Starting training monitor for {self.history_file}")
        logger.info("Close the plot window to stop monitoring")
        
        # Set up animation
        ani = animation.FuncAnimation(self.fig, self.animate, interval=self.update_interval * 1000, 
                                    blit=False, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")

def main():
    parser = argparse.ArgumentParser(description="Real-time training monitor")
    parser.add_argument("--history_file", default="outputs/training_history.json",
                       help="Path to training history JSON file")
    parser.add_argument("--update_interval", type=int, default=5,
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.history_file, args.update_interval)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
