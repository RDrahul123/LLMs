"""
FastAPI LLM Service — Production-ready API server.

Endpoints:
  POST /chat          — Single-turn chat completion
  POST /chat/stream   — SSE streaming chat completion
  POST /chat/multi-turn — Multi-turn conversation
  POST /embed         — Text embeddings
  GET  /health        — Health check
  GET  /models        — List available models

Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import time
import uuid
from typing import AsyncGenerator

import openai
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o"
AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "text-embedding-3-small"]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")

    @validator("role")
    def validate_role(cls, v):
        if v not in ("system", "user", "assistant"):
            raise ValueError("Role must be system, user, or assistant")
        return v


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_items=1, max_items=100)
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: str | list[str] | None = None


class EmbedRequest(BaseModel):
    input: str | list[str] = Field(..., min_length=1)
    model: str = Field(default="text-embedding-3-small")


class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    tokens_used: int
    latency_ms: float


class EmbedResponse(BaseModel):
    id: str
    embeddings: list[list[float]]
    model: str
    dimensions: int
    latency_ms: float


class ModelInfo(BaseModel):
    id: str
    type: str  # chat or embedding


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Service",
    version="1.0.0",
    description="Production-ready LLM API service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_start_time = time.time()

# ---------------------------------------------------------------------------
# Middleware — Logging & Timing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start) * 1000
    print(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({duration_ms:.1f}ms)"
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
    return response


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_client():
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY not configured",
        )
    return client


def get_api_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    # Allow unauthenticated in dev
    if os.environ.get("ENV") == "production":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        uptime_seconds=time.time() - _start_time,
        version="1.0.0",
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    models = [
        ModelInfo(id=m, type="chat" if "gpt" in m else "embedding")
        for m in AVAILABLE_MODELS
    ]
    return ModelsResponse(models=models)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client: openai.AsyncOpenAI = Depends(get_client)):
    start = time.time()
    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[m.dict() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
        )
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    latency = (time.time() - start) * 1000

    return ChatResponse(
        id=response.id,
        content=content,
        model=response.model,
        tokens_used=tokens,
        latency_ms=latency,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, client: openai.AsyncOpenAI = Depends(get_client)):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=[m.dict() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {delta.content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR: {e}]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat/multi-turn", response_model=ChatResponse)
async def chat_multi_turn(
    request: ChatRequest,
    client: openai.AsyncOpenAI = Depends(get_client),
):
    start = time.time()
    messages = [m.dict() for m in request.messages]

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    latency = (time.time() - start) * 1000

    return ChatResponse(
        id=response.id,
        content=content,
        model=response.model,
        tokens_used=tokens,
        latency_ms=latency,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, client: openai.AsyncOpenAI = Depends(get_client)):
    start = time.time()
    try:
        response = await client.embeddings.create(
            model=request.model,
            input=request.input,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    embeddings = [item.embedding for item in response.data]
    latency = (time.time() - start) * 1000
    dimensions = len(embeddings[0]) if embeddings else 0

    return EmbedResponse(
        id=response.id,
        embeddings=embeddings,
        model=response.model,
        dimensions=dimensions,
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
