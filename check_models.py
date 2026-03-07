import requests

API_KEY = "AIzaSyADcFFjrLAbVmCyE66y068D87WpJcnGEcc"
# This endpoint lists every model your key can access in 2026
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url)
if response.status_code == 200:
    models = response.json().get('models', [])
    print("--- COPY THE EXACT NAME BELOW ---")
    for m in models:
        if "generateContent" in m['supportedGenerationMethods']:
            # Look for something like 'models/gemini-3-flash-preview'
            print(f"✅ {m['name']}")
else:
    print(f"Error: {response.status_code} - {response.text}")