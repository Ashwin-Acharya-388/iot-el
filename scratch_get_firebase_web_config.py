import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import requests
import sys

try:
    creds = service_account.Credentials.from_service_account_file(
        'firebase-service-account.json',
        scopes=['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/firebase']
    )
    creds.refresh(Request())
    token = creds.token

    headers = {'Authorization': f'Bearer {token}'}
    resp = requests.get('https://firebase.googleapis.com/v1beta1/projects/iot-el-9509c/webApps', headers=headers)
    apps = resp.json()
    
    if 'apps' in apps and len(apps['apps']) > 0:
        app_id = apps['apps'][0]['appId']
        config_resp = requests.get(f'https://firebase.googleapis.com/v1beta1/projects/iot-el-9509c/webApps/{app_id}/config', headers=headers)
        print(json.dumps(config_resp.json(), indent=2))
    else:
        print("Still no apps found.")
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
