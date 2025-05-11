import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ANSI color codes
COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "reset": "\033[0m"
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages"""
    
    def format(self, record):
        # Add color based on log level
        if record.levelno >= logging.ERROR:
            color = COLORS["red"]
        elif record.levelno >= logging.WARNING:
            color = COLORS["yellow"]
        elif record.levelno >= logging.INFO:
            color = COLORS["green"]
        else:  # DEBUG
            color = COLORS["blue"]
            
        # Format the message with color
        record.msg = f"{color}{record.msg}{COLORS['reset']}"
        return super().format(record)

class SystemLogger:
    """System-wide logger for the obstacle avoidance system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger("obstacle_avoidance")
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = ColoredFormatter(
            '%(levelname)s: %(message)s'
        )
        
        # Create handlers for each log level
        self.setup_log_handlers(file_formatter, console_formatter)
        
        # Component-specific loggers
        self.detection_logger = self.logger.getChild("detection")
        self.image_logger = self.logger.getChild("image_processing")
        self.visualization_logger = self.logger.getChild("visualization")
        self.environment_logger = self.logger.getChild("environment")

    def setup_log_handlers(self, file_formatter, console_formatter):
        """Set up handlers for different log levels"""
        # Debug log file
        debug_file = self.log_dir / "debug.log"
        debug_handler = logging.FileHandler(debug_file)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        self.logger.addHandler(debug_handler)
        
        # Info log file
        info_file = self.log_dir / "info.log"
        info_handler = logging.FileHandler(info_file)
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(file_formatter)
        self.logger.addHandler(info_handler)
        
        # Warning log file
        warning_file = self.log_dir / "warning.log"
        warning_handler = logging.FileHandler(warning_file)
        warning_handler.setLevel(logging.WARNING)
        warning_handler.setFormatter(file_formatter)
        self.logger.addHandler(warning_handler)
        
        # Error log file
        error_file = self.log_dir / "error.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self, component: str) -> logging.Logger:
        """Get a component-specific logger"""
        return self.logger.getChild(component)

    def log_error(self, component: str, error: Exception, context: Optional[str] = None):
        """Log an error with context"""
        error_msg = f"{component} - {str(error)}"
        if context:
            error_msg += f" - Context: {context}"
        self.logger.error(error_msg, exc_info=True)

    def log_warning(self, component: str, message: str):
        """Log a warning message"""
        self.logger.warning(f"{component} - {message}")

    def log_info(self, component: str, message: str):
        """Log an info message"""
        self.logger.info(f"{component} - {message}")

    def log_debug(self, component: str, message: str):
        """Log a debug message"""
        self.logger.debug(f"{component} - {message}")

# Create global logger instance
system_logger = SystemLogger() 