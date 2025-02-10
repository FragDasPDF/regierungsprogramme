import json
from datetime import datetime

def save_comparison_results(results, filename_prefix="comparison"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/results/{filename_prefix}_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    return filename 