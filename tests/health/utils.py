from __future__ import annotations

import logging
from typing import Optional, Callable, Any
from datetime import datetime
from functools import wraps


def timed_check(func: Callable) -> Callable:
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        
        if hasattr(result, 'duration') and result.duration is None:
            result.duration = duration
        
        return result
    
    return wrapper


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
            return getattr(module, package)
        else:
            return __import__(module_name)
    except ImportError as e:
        logging.warning(f"Failed to import {module_name}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error importing {module_name}: {e}")
        return None


def format_duration(seconds: float) -> str:
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def print_section_header(title: str, width: int = 60, char: str = "-"):
    
    print(f"\n{title}")
    print(char * width)


def print_subsection(title: str, width: int = 60):
    
    print(f"\n{title}")
    print("." * width)
