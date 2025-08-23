# Save report if path provided
import numpy as np
import pandas as pd

def make_json_safe(obj):
    """Recursively convert numpy/pandas types to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    else:
        return obj