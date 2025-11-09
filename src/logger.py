"""
Experiment logging and tracking utility.
Supports both WandB and TensorBoard backends.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from config import LOGGING_CONFIG, RESULTS_DIR


class ExperimentLogger:
    """Logger for experiment metrics and configurations.
    
    Supports WandB and TensorBoard backends for experiment tracking.
    """
    
    def __init__(self, 
                 project_name: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 backend: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize experiment logger.
        
        Args:
            project_name: Project name (default from config)
            experiment_name: Experiment name/run name
            backend: 'wandb' or 'tensorboard' (default from config)
            config: Configuration dictionary to log
        """
        self.backend = backend or LOGGING_CONFIG['backend']
        self.project_name = project_name or LOGGING_CONFIG['project_name']
        self.experiment_name = experiment_name or 'run'
        self.config = config or {}
        
        self.writer = None
        self.run = None
        
        # Initialize backend
        self._initialize_backend()
        
        # Log configuration
        if self.config:
            self.log_config(self.config)
    
    def _initialize_backend(self) -> None:
        """Initialize logging backend (WandB or TensorBoard)."""
        if self.backend == 'wandb':
            try:
                import wandb
                self.run = wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=self.config
                )
                print(f"WandB initialized: {self.project_name}/{self.experiment_name}")
            except ImportError:
                print("WARNING: wandb not installed. Falling back to local logging.")
                self.backend = 'local'
        
        elif self.backend == 'tensorboard':
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = RESULTS_DIR / 'runs' / self.experiment_name
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(log_dir))
                print(f"TensorBoard initialized: {log_dir}")
            except ImportError:
                print("WARNING: tensorboard not installed. Falling back to local logging.")
                self.backend = 'local'
        
        else:
            print(f"Using local logging (backend: {self.backend})")
            self.backend = 'local'
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step/iteration number
        """
        if self.backend == 'wandb' and self.run:
            import wandb
            wandb.log({name: value}, step=step)
        
        elif self.backend == 'tensorboard' and self.writer:
            if step is not None:
                self.writer.add_scalar(name, value, step)
            else:
                self.writer.add_scalar(name, value)
        
        else:
            # Local logging
            log_file = RESULTS_DIR / f"{self.experiment_name}_metrics.jsonl"
            with open(log_file, 'a') as f:
                entry = {'name': name, 'value': value, 'step': step}
                f.write(json.dumps(entry) + '\n')
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step/iteration number
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration dictionary.
        
        Args:
            config: Configuration dictionary
        """
        if self.backend == 'wandb' and self.run:
            import wandb
            wandb.config.update(config)
        
        elif self.backend == 'tensorboard' and self.writer:
            # TensorBoard doesn't have direct config logging
            # Write to hparams or save as text
            config_str = json.dumps(config, indent=2)
            self.writer.add_text('config', config_str)
        
        else:
            # Local logging
            config_file = RESULTS_DIR / f"{self.experiment_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    def log_model(self, model_path: Path, name: Optional[str] = None) -> None:
        """Log a model artifact.
        
        Args:
            model_path: Path to the model file
            name: Artifact name (optional)
        """
        if self.backend == 'wandb' and self.run:
            import wandb
            artifact_name = name or model_path.stem
            artifact = wandb.Artifact(artifact_name, type='model')
            artifact.add_file(str(model_path))
            self.run.log_artifact(artifact)
        
        else:
            # For tensorboard/local, just log the path
            print(f"Model saved: {model_path}")
    
    def log_table(self, name: str, data: Dict[str, list]) -> None:
        """Log a table of data.
        
        Args:
            name: Table name
            data: Dictionary with column names as keys and lists as values
        """
        if self.backend == 'wandb' and self.run:
            import wandb
            table = wandb.Table(data=data)
            wandb.log({name: table})
        
        else:
            # Save as JSON for local/tensorboard
            table_file = RESULTS_DIR / f"{self.experiment_name}_{name}.json"
            with open(table_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def finish(self) -> None:
        """Finish logging and close connections."""
        if self.backend == 'wandb' and self.run:
            import wandb
            wandb.finish()
            print("WandB run finished")
        
        elif self.backend == 'tensorboard' and self.writer:
            self.writer.close()
            print("TensorBoard writer closed")
        
        else:
            print("Local logging finished")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_logger(experiment_name: str, config: Optional[Dict] = None) -> ExperimentLogger:
    """Create and return an experiment logger.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary to log
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(
        experiment_name=experiment_name,
        config=config
    )
