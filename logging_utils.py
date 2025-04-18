import logging
from datetime import datetime

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_interaction(query, response, query_type, duration_ms=0, error=None):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "query_type": query_type,
        "response_length": len(str(response)),
        "duration_ms": duration_ms,
        "error": str(error) if error else None
    }
    logging.info(f"INTERACTION: {log_data}")