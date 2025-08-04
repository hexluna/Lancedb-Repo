from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import asyncio
import functools
import traceback
import json
import psutil
import os

# Import your existing functions
from query_test import query_faiss
from dotenv import load_dotenv


process = psutil.Process(os.getpid())
mem_start = process.memory_info().rss / (1024 * 1024)
print(f"🚀 Memory at server start: {mem_start:.2f} MB")

load_dotenv()

app = FastAPI(
    title="SIT Chatbot - OpenAI Compatible API",
    description="Singapore Institute of Technology Chatbot with OpenAI API compatibility",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# OpenAI-compatible models
class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str = "sit-chatbot"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List] = None
    # ElevenLabs custom parameters
    extra_body: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


# Global initialization flag
system_initialized = True  # Set to True since imports handle initialization


@app.on_event("startup")
async def startup_event():
    print("🚀 SIT Chatbot Server Starting...")
    print("✅ Server ready for ElevenLabs integration!")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log incoming request
    print(f"\n🔍 INCOMING REQUEST:")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")

    # Get request body for POST requests
    if request.method == "POST":
        body = await request.body()
        if body:
            try:
                json_body = json.loads(body.decode())
                print(f"Body: {json.dumps(json_body, indent=2)}")
            except:
                print(f"Raw Body: {body}")

    response = await call_next(request)

    process_time = time.time() - start_time
    print(f"⏱️ Request processed in {process_time:.2f}s")
    print(f"Status: {response.status_code}")

    return response


@app.get("/")
async def root():
    return {
        "message": "SIT Chatbot API - OpenAI Compatible",
        "status": "running",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
            "test": "/test",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "SIT Chatbot API is operational",
        "timestamp": int(time.time())
    }


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "Test successful!",
        "openai_format": {
            "id": f"test-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "sit-chatbot",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response in OpenAI format."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }
    }


# Main OpenAI-compatible endpoint
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    print(f"\n🎯 CHAT COMPLETION REQUEST RECEIVED")
    print(f"Model: {request.model}")
    print(f"Messages: {len(request.messages)}")
    print(f"Temperature: {request.temperature}")
    print(f"Stream: {request.stream}")
    print(f"Stream options: {request.stream_options}")
    print(f"Extra body: {request.extra_body}")

    try:
        # Extract the user's message (last message with role "user")
        user_message = ""
        print(f"\n📝 ANALYZING MESSAGES:")
        for i, message in enumerate(request.messages):
            print(f"  {i + 1}. Role: {message.role}, Content: {message.content[:100]}...")
            if message.role == "user":
                user_message = message.content

        if not user_message.strip():
            print("❌ ERROR: No user message found")
            raise HTTPException(status_code=400, detail="No user message found in request")

        print(f"\n🔍 PROCESSING USER MESSAGE: {user_message}")


        # Process the query using your existing system
        start_time = time.time()
        print(f"⏳ Starting SIT chatbot processing...")

        loop = asyncio.get_event_loop()
        response_content = await loop.run_in_executor(
            None,
            functools.partial(query_faiss, user_message)
        )

        processing_time = time.time() - start_time
        print(f"✅ SIT chatbot processing completed in {processing_time:.2f}s")
        print(f"📤 Response length: {len(response_content)} characters")
        print(f"📝 Response preview: {response_content[:200]}...")

        # Calculate token usage (rough estimation)
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = len(response_content.split())
        total_tokens = prompt_tokens + completion_tokens

        # Handle streaming vs non-streaming
        if request.stream:
            print("📡 STREAMING RESPONSE TO ELEVENLABS")

            async def generate_stream():
                try:
                    # Send the content as a single chunk
                    chunk = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": response_content
                            },
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                    # Send usage info if requested
                    if request.stream_options and request.stream_options.include_usage:
                        usage_chunk = {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens
                            }
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"

                    yield "data: [DONE]\n\n"

                except Exception as e:
                    print(f"❌ Streaming error: {str(e)}")
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "internal_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8"
                }
            )

        else:
            # Non-streaming response
            response_data = {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }

            print(f"\n📋 NON-STREAMING RESPONSE STRUCTURE:")
            print(f"ID: {response_data['id']}")
            print(f"Object: {response_data['object']}")
            print(f"Created: {response_data['created']}")
            print(f"Model: {response_data['model']}")
            print(f"Usage: {response_data['usage']}")
            print(f"Content length: {len(response_data['choices'][0]['message']['content'])}")

            print(f"\n📤 SENDING NON-STREAMING RESPONSE TO ELEVENLABS")
            return response_data

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n❌ ERROR in chat completion:")
        print(f"Error: {error_msg}")
        print(f"Traceback: {error_trace}")

        # Return proper error response
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": error_msg,
                "type": "server_error"
            }
        )


# Alternative endpoint for direct testing (your original format)
@app.post("/chat")
async def chat_endpoint(request: dict):
    """Direct chat endpoint for testing"""
    print(f"\n🔄 DIRECT CHAT REQUEST: {request}")

    try:
        message = request.get("message", "")
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        start_time = time.time()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(query_faiss, message)
        )

        result = {
            "response": response,
            "processing_time": round(time.time() - start_time, 2),
            "timestamp": int(time.time())
        }

        print(f"✅ Direct chat response: {result}")
        return result

    except Exception as e:
        print(f"❌ Direct chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting SIT Chatbot Server (OpenAI Compatible)...")
    print("🔧 Debug mode enabled - all requests will be logged")
    print("🌐 ElevenLabs integration ready with streaming support")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )