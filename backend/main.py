"""Verdict API."""
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from models.schemas import VerifyRequest, VerifyResponse
from services.llm_client import LLMError
from services.pipeline import JudgeParseError, build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verdict.api")


def create_app(pipeline_factory: Optional[Callable] = None, api_key: Optional[str] = None) -> FastAPI:
    """`pipeline_factory` returns a VerificationPipeline (injectable for tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline_factory is not None:
            app.state.pipeline = pipeline_factory()
        else:
            from config import load_settings
            settings = load_settings()
            app.state.api_key = settings.api_key
            app.state.pipeline = build_pipeline(settings)
        yield

    app = FastAPI(title="Verdict API",
                  description="Evidence-Based AI Claim Verification Platform",
                  version="1.1.0", lifespan=lifespan)
    app.state.api_key = api_key

    def require_key(request: Request, x_api_key: Optional[str] = Header(default=None)):
        expected = getattr(request.app.state, "api_key", None)
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    @app.get("/")
    def root():
        return {"message": "Verdict API is running"}

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok" if getattr(request.app.state, "pipeline", None) else "starting"}

    @app.post("/api/verify", response_model=VerifyResponse, dependencies=[Depends(require_key)])
    def verify_claim(body: VerifyRequest, request: Request):
        claim = body.claim.strip()
        if not claim:
            raise HTTPException(status_code=422, detail="Claim cannot be empty.")
        try:
            return request.app.state.pipeline.verify(claim)
        except LLMError as exc:
            logger.error("LLM failure: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream language model unavailable.")
        except JudgeParseError as exc:
            logger.error("Judge parse failure: %s", exc)
            raise HTTPException(status_code=502, detail="Judge returned an invalid response.")
        except Exception:
            logger.exception("Unhandled error while verifying claim")
            raise HTTPException(status_code=500, detail="Internal server error.")

    return app


app = create_app()
