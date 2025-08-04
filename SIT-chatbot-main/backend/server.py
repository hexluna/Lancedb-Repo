# backend/server.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to handle imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stt_elevenlabs import transcribe_audio

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes should be defined before static file handling
@app.get("/api/signed-url")
async def get_signed_url():
    # Use SIT otter assistant agent ID
    agent_id = os.getenv("AGENT_ID")
    xi_api_key = os.getenv("XI_API_KEY")
    
    if not agent_id or not xi_api_key:
        raise HTTPException(status_code=500, detail="Missing environment variables")
    
    url = f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={agent_id}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers={"xi-api-key": xi_api_key})
            response.raise_for_status()
            data = response.json()
            return {"signedUrl": data["signed_url"]}

        except httpx.HTTPStatusError as e:
            print("❌ ElevenLabs API returned an error:")
            print(f"Status code: {e.response.status_code}")
            print(f"Body: {e.response.text}")
            raise HTTPException(status_code=500, detail="Failed to get signed URL")

        except Exception as e:
            print("❌ Unexpected error:", str(e))
            raise HTTPException(status_code=500, detail="Unexpected error occurred")
    
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = await client.get(
    #             url,
    #             headers={"xi-api-key": xi_api_key}
    #         )
    #         response.raise_for_status()
    #         data = response.json()
    #         return {"signedUrl": data["signed_url"]}
    #
    #     except httpx.HTTPError:
    #         raise HTTPException(status_code=500, detail="Failed to get signed URL")


#API route for getting Agent ID, used for public agents
@app.get("/api/getAgentId")
def get_unsigned_url():
    agent_id = os.getenv("AGENT_ID")
    return {"agentId": agent_id}

# API endpoint for speech-to-text transcription
@app.post("/api/transcribe")
async def transcribe_speech(file: UploadFile = File(...), language: str = Form("en")):
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the uploaded file temporarily
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Transcribe the audio using the ElevenLabs API
        transcript = transcribe_audio(file_path, language)
        
        # Clean up the temporary file
        os.remove(file_path)
        
        return JSONResponse(content={"text": transcript})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {str(e)}"}
        )

# Mount static files for specific assets (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="dist"), name="static")

# Serve index.html for root path
@app.get("/")
async def serve_root():
    return FileResponse("dist/index.html")