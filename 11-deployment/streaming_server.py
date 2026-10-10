"""
Streaming Server — SSE streaming for LLM responses.

Implements:
  - FastAPI StreamingResponse
  - Server-Sent Events (SSE) format
  - Token-by-token streaming from OpenAI
  - Streaming with function calling
  - Client-side SSE consumption example
  - Streaming with conversation history

Run: python streaming_server.py
"""

import asyncio
import json
import os
import time
from typing import AsyncGenerator

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str
    content: str


class StreamRequest(BaseModel):
    messages: list[Message] = Field(..., min_items=1)
    model: str = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    stream: bool = Field(default=True)


class FunctionCallRequest(BaseModel):
    messages: list[Message] = Field(..., min_items=1)
    model: str = Field(default=DEFAULT_MODEL)
    functions: list[dict] = Field(default_factory=list)
    stream: bool = Field(default=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Streaming LLM Server")

client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------------------------------------------------------------------------
# SSE Streaming Endpoint
# ---------------------------------------------------------------------------


@app.post("/stream")
async def stream_chat(request: StreamRequest):
    """Token-by-token streaming with SSE format."""

    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    async def event_generator() -> AsyncGenerator[str, None]:
        request_id = f"req_{int(time.time() * 1000)}"

        # Send start event
        yield f"event: start\ndata: {json.dumps({'request_id': request_id, 'model': request.model})}\n\n"

        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=[m.dict() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            token_count = 0
            async for chunk in stream:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                if delta.content:
                    token_count += 1
                    yield f"event: token\ndata: {json.dumps({'content': delta.content, 'token_index': token_count})}\n\n"

                if finish_reason:
                    yield f"event: done\ndata: {json.dumps({'finish_reason': finish_reason, 'total_tokens': token_count})}\n\n"

        except openai.RateLimitError:
            yield f"event: error\ndata: {json.dumps({'error': 'Rate limit exceeded'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Streaming with Function Calling
# ---------------------------------------------------------------------------


@app.post("/stream/functions")
async def stream_with_functions(request: FunctionCallRequest):
    """Streaming with function/tool calling support."""

    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    if not request.functions:
        raise HTTPException(status_code=400, detail="No functions provided")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=[m.dict() for m in request.messages],
                functions=request.functions,
                stream=True,
            )

            function_call = None
            async for chunk in stream:
                delta = chunk.choices[0].delta

                # Handle function call chunks
                if delta.function_call:
                    if function_call is None:
                        function_call = {"name": "", "arguments": ""}
                    if delta.function_call.name:
                        function_call["name"] += delta.function_call.name
                    if delta.function_call.arguments:
                        function_call["arguments"] += delta.function_call.arguments
                    yield f"event: function_call\ndata: {json.dumps(delta.function_call, default=str)}\n\n"

                # Handle text content
                if delta.content:
                    yield f"event: token\ndata: {json.dumps({'content': delta.content})}\n\n"

            # Send final function call if present
            if function_call:
                yield f"event: function_call_complete\ndata: {json.dumps(function_call)}\n\n"
            else:
                yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# Streaming with Conversation History
# ---------------------------------------------------------------------------


@app.post("/stream/conversation")
async def stream_conversation(request: StreamRequest):
    """Streaming endpoint that handles conversation history properly."""

    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    async def event_generator() -> AsyncGenerator[str, None]:
        full_response = []

        try:
            stream = await client.chat.completions.create(
                model=request.model,
                messages=[m.dict() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response.append(delta.content)
                    yield f"data: {delta.content}\n\n"

            # Send metadata as final event
            metadata = {
                "total_chars": len("".join(full_response)),
                "message_count": len(request.messages),
            }
            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Non-streaming fallback
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(request: StreamRequest):
    """Non-streaming chat endpoint for comparison."""

    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    try:
        response = await client.chat.completions.create(
            model=request.model,
            messages=[m.dict() for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Client-side SSE Consumption Example
# ---------------------------------------------------------------------------

CLIENT_EXAMPLE = """
// JavaScript SSE Client Example
const eventSource = new EventSource('/stream', { method: 'POST' });

// For POST requests, use fetch with ReadableStream:
async function streamChat(messages) {
    const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages, model: 'gpt-4o' }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop(); // Keep incomplete line

        for (const line of lines) {
            if (line.startsWith('event: ')) {
                const eventType = line.slice(7);
                console.log('Event:', eventType);
            } else if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') {
                    console.log('Stream complete');
                    return;
                }
                try {
                    const parsed = JSON.parse(data);
                    if (parsed.content) {
                        process.stdout.write(parsed.content);
                    }
                } catch {
                    process.stdout.write(data);
                }
            }
        }
    }
}

// Python SSE Client Example
# import httpx
# import json
#
# async def stream_chat(messages):
#     async with httpx.AsyncClient() as client:
#         async with client.stream(
#             "POST",
#             "http://localhost:8000/stream",
#             json={"messages": messages, "model": "gpt-4o"},
#         ) as response:
#             async for line in response.aiter_lines():
#                 if line.startswith("data: "):
#                     data = line[6:]
#                     if data == "[DONE]":
#                         break
#                     try:
#                         parsed = json.loads(data)
#                         if "content" in parsed:
#                             print(parsed["content"], end="", flush=True)
#                     except json.JSONDecodeError:
#                         pass
"""


@app.get("/client-example")
async def client_example():
    """Returns client-side consumption examples."""
    return {"javascript_and_python": CLIENT_EXAMPLE}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("Starting streaming server on http://localhost:8001")
    print("Endpoints:")
    print("  POST /stream              — Token-by-token SSE streaming")
    print("  POST /stream/functions     — Streaming with function calling")
    print("  POST /stream/conversation  — Streaming with history")
    print("  POST /chat                — Non-streaming fallback")
    print("  GET  /client-example       — Client code examples")
    uvicorn.run(app, host="0.0.0.0", port=8001)
