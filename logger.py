"""
Simple logging system with verbosity levels
Reduces console clutter
"""

class Logger:
    """Simple logger with verbosity control"""
    
    # Logging levels
    QUIET = 0      # Only critical errors
    NORMAL = 1     # Important events only
    VERBOSE = 2    # Detailed logging
    DEBUG = 3      # Everything
    
    def __init__(self, level=NORMAL):
        self.level = level
        
    def debug(self, message):
        """Debug level logging"""
        if self.level >= self.DEBUG:
            print(f"[DEBUG] {message}")
            
    def info(self, message):
        """Info level logging"""
        if self.level >= self.VERBOSE:
            print(message)
            
    def important(self, message):
        """Important events (normal level)"""
        if self.level >= self.NORMAL:
            print(message)
            
    def critical(self, message):
        """Critical events (always shown)"""
        print(message)
        
    def error(self, message):
        """Errors (always shown)"""
        print(f"❌ {message}")


# Global logger instance
_logger = Logger(level=Logger.NORMAL)

def set_verbosity(level):
    """Set global logging verbosity"""
    global _logger
    _logger.level = level
    
def log_debug(message):
    """Log debug message"""
    _logger.debug(message)
    
def log_info(message):
    """Log info message"""
    _logger.info(message)
    
def log(message):
    """Log important message"""
    _logger.important(message)
    
def log_critical(message):
    """Log critical message"""
    _logger.critical(message)
    
def log_error(message):
    """Log error message"""
    _logger.error(message)

