import requests
import json

class GeminiService:
    def __init__(self, api_key):
        self.api_key = api_key
        # UPDATED FOR 2026: 1.5 is retired. 
        # Use 'gemini-3-flash-preview' for the best balance of speed and reasoning.
        # Alternatively, use 'gemini-3.1-flash-lite-preview' for massive volume.
        self.model_name = "gemini-3-flash-preview" 
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

    def generate_answer(self, query, context_chunks):
        context_text = "\n\n".join([c.payload['content'] for c in context_chunks])
        
        # New 2026 Prompt structure optimized for Gemini 3
        prompt = f"""
        Answer the question using ONLY the provided context.
        
        CONTEXT:
        {context_text}
        
        QUESTION:
        {query}
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,  # Keep it precise for your ML notes
                "maxOutputTokens": 1000
            }
        }
        
        try:
            response = requests.post(self.url, json=payload)
            data = response.json()
            
            if response.status_code == 429:
                return "⚠️ QUOTA EXCEEDED: Please wait 60 seconds."
            if response.status_code != 200:
                return f"Error {response.status_code}: {data.get('error', {}).get('message')}"
            
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            return f"System Error: {str(e)}"