import json

def format_json(data):
    """Devuelve un JSON formateado y legible."""
    return json.dumps(data, indent=4)
