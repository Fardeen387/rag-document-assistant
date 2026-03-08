import os
import uvicorn

if __name__ == "__main__":
    # Get port from Railway's environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run the app
    uvicorn.run("app:app", host="0.0.0.0", port=port)