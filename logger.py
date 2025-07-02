import enum
import sys
import datetime
import inspect

class LogLevel(enum.Enum):
    INFO = 'INFO'
    ERROR = 'ERROR'
    DEBUG = 'DEBUG'
    SUCCESS = 'SUCCESS'

def log(level: LogLevel, message: str):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename.split("\\")[-1]
    lineno = frame.f_lineno
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{level.value}] [{timestamp}] [{filename}/{lineno}] - {message}", file=sys.stdout)

def log_info(message: str):
    log(LogLevel.INFO, message)

def log_error(message: str):
    log(LogLevel.ERROR, message)

def log_debug(message: str):
    log(LogLevel.DEBUG, message)

def log_success(message: str):
    log(LogLevel.SUCCESS, message)
