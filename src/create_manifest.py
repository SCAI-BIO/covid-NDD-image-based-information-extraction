import pandas as pd
import hashlib
import requests
from datetime import datetime

def create_manifest():
    # Load your URLs
    df = pd.read_excel('data/URL_relevance_analysis/Final_Relevant_URLs.xlsx')
    
    manifest = []
    for idx, row in df.iterrows():
        url = row['URL']  # adjust column name
        
        # Try to fetch and hash
        try:
            response = requests.get(url, timeout=10)
            http_status = response.status_code
            
            if http_status == 200:
                sha256 = hashlib.sha256(response.content).hexdigest()
            else:
                sha256 = "N/A"
        except:
            http_status = "Error"
            sha256 = "N/A"
        
        manifest.append({
            'Image_Number': idx + 1,
            'Original_URL': url,
            'GitHub_URL': row.get('GitHub_URL', 'N/A'),  # if available
            'HTTP_Status': http_status,
            'SHA256': sha256,
            'Timestamp': datetime.now().isoformat()
        })
    
    pd.DataFrame(manifest).to_csv('data/image_manifest.csv', index=False)

if __name__ == "__main__":
    create_manifest()