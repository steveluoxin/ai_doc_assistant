from app import app

# Vercel serverless entrypoint expects an ASGI app exposed as `app`.
# We import the FastAPI app and re-export it.

# This handler is also used by Vercel's Python runtime when running
# locally with `vercel dev`.
