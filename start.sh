#!/bin/sh
# Use the port assigned by Railway, or default to 8000
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}