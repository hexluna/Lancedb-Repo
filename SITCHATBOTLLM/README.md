Here's a clean and simple README.md for your SIT Chatbot Agent:

# SIT Chatbot Agent

A simple chatbot server with public HTTPS access via ngrok.

## Setup

### Install Dependencies
You will first need to create a python virtual environment and then install the required dependencies in the virtual environment via the command below.

```
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY="your-secret-api-key-here"
```

You will need an OpenAI API Key from the OpenAI console.

## Running the Agent

### Start the Server
```
python server.py
```

### Expose Public URL
You will need to install a new 

In a new terminal window:
```
ngrok http 8000
```

## Usage

After running ngrok, you'll get a public HTTPS URL (e.g., `https://random-string.ngrok-free.app`).

The main endpoint is available at:
```
POST /v1/chat/completions
```

You will then create a new agent on the Conversational AI Web App, where you will create your bot and configure the Custom LLM model via the url given above.  
Ensure the URL for the custom LLM is without the /chat/completions, Eleven Labs will append it. 
