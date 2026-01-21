import sys
import os
import time
from ultralytics import YOLO
from datetime import datetime

# --- Configuration ---
PROJECT_NAME = "pencil_detection"
DATA_YAML = "data.yaml"
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 8
WORKERS = 4
PATIENCE = 10
DEVICE = 0

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.table import Table
    from rich.console import Console
    from rich import box
    
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Running in standard mode.")


class TrainingDashboard:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.console = Console()
        self.layout = Layout()
        
        # Create layout structure
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # Split body into progress and stats
        self.layout["body"].split_row(
            Layout(name="progress", ratio=1), 
            Layout(name="stats", ratio=1)
        )
        
        # Create progress tracker
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        
        self.epoch_task = self.overall_progress.add_task(
            "[cyan]Total Progress", total=self.total_epochs
        )
        
        # Stats table
        self.stats_table = Table(box=box.SIMPLE_HEAD)
        self.stats_table.add_column("Metric", style="cyan")
        self.stats_table.add_column("Value", style="magenta")
        
        self.update_layout()
    
    def update_layout(self):
        # Header
        self.layout["header"].update(
            Panel(
                "[bold white on blue] YOLOv8 Training Dashboard - Pencil Detection [/]",
                style="bold white on blue",
                border_style="blue"
            )
        )
        
        # Progress section
        self.layout["progress"].update(
            Panel(self.overall_progress, title="Training Status", border_style="green")
        )
        
        # Stats section
        self.layout["stats"].update(
            Panel(self.stats_table, title="Current Metrics", border_style="yellow")
        )
        
        # Footer
        self.layout["footer"].update(
            Panel(
                "[dim white]Press Ctrl+C to stop training[/]", 
                style="dim white", 
                border_style="dim white"
            )
        )
    
    def update_progress(self, epoch):
        self.overall_progress.update(self.epoch_task, completed=epoch + 1)
    
    def update_metrics(self, metrics):
        self.stats_table = Table(box=box.SIMPLE_HEAD)
        self.stats_table.add_column("Metric", style="cyan")
        self.stats_table.add_column("Value", style="magenta")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self.stats_table.add_row(key, f"{value:.4f}")
            else:
                self.stats_table.add_row(key, str(value))
        
        self.layout["stats"].update(
            Panel(self.stats_table, title="Current Metrics", border_style="yellow")
        )


def on_train_epoch_end(trainer):
    """Callback function executed at the end of each training epoch"""
    if dashboard:
        metrics = {
            "mAP50": trainer.metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": trainer.metrics.get("metrics/mAP50-95(B)", 0),
            "Precision": trainer.metrics.get("metrics/precision(B)", 0),
            "Recall": trainer.metrics.get("metrics/recall(B)", 0)
        }
        
        if len(trainer.loss_items) > 0:
            metrics["Box Loss"] = float(trainer.loss_items[0])
        if len(trainer.loss_items) > 1:
            metrics["Cls Loss"] = float(trainer.loss_items[1])
        if len(trainer.loss_items) > 2:
            metrics["DFL Loss"] = float(trainer.loss_items[2])
        
        dashboard.update_metrics(metrics)
        dashboard.update_progress(trainer.epoch)


if __name__ == "__main__":
    
    # Training arguments
    train_args = {
        "data": DATA_YAML,
        "imgsz": IMG_SIZE,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "name": PROJECT_NAME,
        "patience": PATIENCE,
        "workers": WORKERS,
        "verbose": True,
    }
    
    # Load model
    model = YOLO("yolov8n.pt")
    
    if RICH_AVAILABLE:
        dashboard = TrainingDashboard(total_epochs=EPOCHS)
        
        # Add callbacks to model
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        print("\n" + "="*50)
        print(f"Starting YOLOv8 Training: {PROJECT_NAME}")
        print("="*50 + "\n")
        
        try:
            with Live(dashboard.layout, refresh_per_second=4, screen=True):
                model.train(**train_args)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            print("Model checkpoint saved.")
        except Exception as e:
            print(f"\nError during training: {e}")
            model.train(**train_args)
    else:
        print("\n" + "="*50)
        print(f"Starting YOLOv8 Training: {PROJECT_NAME}")
        print("="*50)
        print(f"Image Size: {IMG_SIZE}")
        print(f"Epochs: {EPOCHS}")
        print(f"Batch Size: {BATCH_SIZE}")
        print("="*50 + "\n")
        
        model.train(**train_args)
    
    # Print completion message
    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50)
    print(f"\nExporting Model to ONNX format...")
    print("="*50 + "\n")
    
    # Get best model path
    best_model_path = os.path.join("runs", "detect", PROJECT_NAME, "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        print(f"Best model found at: {best_model_path}")
        
        # Export model
        try:
            success = model.export(format="onnx")
            print(f"Export completed: {success}")
        except Exception as e:
            print(f"Export functionality encountered an error: {e}")
    else:
        print(f"Best model not found at expected path: {best_model_path}")
        print("Exporting current model state...")
        try:
            success = model.export(format="onnx")
            print(f"Export completed: {success}")
        except Exception as e:
            print(f"Export functionality encountered an error: {e}")
