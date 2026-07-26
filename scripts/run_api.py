"""Run the FastAPI service backing scripts/app.py (Streamlit UI).

Usage:
    uv run python scripts/run_api.py
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("football_rag.api.main:app", host="127.0.0.1", port=8000)
