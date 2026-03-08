import requests
import json

class GeminiService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = "gemini-3-flash-preview"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        self.stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent?alt=sse&key={self.api_key}"

    def _build_payload(self, query, context_chunks):
        context_text = "\n\n".join([c.payload['content'] for c in context_chunks])
        prompt = f"""Answer the question using ONLY the provided context.

CONTEXT:
{context_text}

QUESTION:
{query}"""
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000
            }
        }

    def generate_answer(self, query, context_chunks):
        """Standard (non-streaming) response."""
        payload = self._build_payload(query, context_chunks)
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

    def stream_answer(self, query, context_chunks):
        """Streaming response — yields text chunks as they arrive."""
        payload = self._build_payload(query, context_chunks)
        try:
            response = requests.post(self.stream_url, json=payload, stream=True)
            if response.status_code == 429:
                yield "⚠️ QUOTA EXCEEDED: Please wait 60 seconds."
                return
            if response.status_code != 200:
                data = response.json()
                yield f"Error {response.status_code}: {data.get('error', {}).get('message')}"
                return

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        json_str = line[6:]  # strip "data: "
                        if json_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(json_str)
                            text = chunk['candidates'][0]['content']['parts'][0]['text']
                            if text:
                                yield text
                        except (KeyError, json.JSONDecodeError):
                            continue
        except Exception as e:
            yield f"System Error: {str(e)}"