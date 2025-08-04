# SIT Chatbot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SIT Chatbot with friendly otter assistant.

This project is a chatbot application designed for SIT, featuring a friendly otter assistant. It utilizes a Node.js backend for handling API requests and a Python backend for AI functionalities, with a frontend built using webpack.

## Table of Contents

* [About The Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [License](#license)

## About The Project

The SIT Chatbot aims to provide an interactive and engaging experience for users seeking information or assistance related to SIT. The friendly otter assistant adds a unique touch to the user interface.

Key features:
* Interactive chat interface
* Dual backend system (Node.js and Python)
* Webpack for frontend asset bundling

## Built With

This project utilizes the following major frameworks and libraries:

**Frontend:**
* HTML
* CSS
* JavaScript

**Backend (Node.js):**
* Express.js
* @elevenlabs/client

**Backend (Python):**
* FastAPI
* Uvicorn

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have the following software installed on your system:
* Node.js and npm (Node Package Manager)
  ```bash
  # Check if installed
  node -v
  npm -v
  ```
* Python and pip (Python Package Installer)
  ```bash
  # Check if installed
  python --version
  pip --version
  ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Finance-LLMs/SIT-chatbot.git
    cd SIT-chatbot
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project. You can copy the example or create it manually.
    ```bash
    # For Windows (using Command Prompt or PowerShell)
    copy .env.example .env # If you have an example file
    notepad .env          # To create and edit a new .env file
    ```
    ```bash
    # For Linux/macOS
    cp .env.example .env # If you have an example file
    vim .env             # To create and edit a new .env file
    ```
    Add necessary environment variables to this file (e.g., API keys, database URIs).  Refer to the specific backend documentation or code for required variables.

3.  **Install frontend dependencies:**
    ```bash
    npm install
    ```

4.  **Install backend dependencies (Python):**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are a couple of ways to run the application, depending on whether you want to run the Node.js backend or the Python backend.

### Running with Node.js Backend

1.  **Build the frontend assets:**
    ```bash
    npm run build
    ```
2.  **Start the Node.js backend server:**
    ```bash
    npm run start:backend
    ```
    Alternatively, to build and start the Node.js backend together:
    ```bash
    npm start
    ```

### Running with Python Backend (using Uvicorn)

1.  **Build the frontend assets (if not already done):**
    ```bash
    npm run build
    ```
2.  **Start the Python backend server:**
    ```bash
    npm run start:python
    ```
    This command will typically run: `uvicorn backend.server:app --reload --port 3000`

### Running in Development Mode (Node.js backend with Webpack Dev Server)

For a development environment with hot reloading for the frontend and concurrent backend execution:
```bash
npm run dev
```
This command uses `concurrently` to run both the Node.js backend (`npm run start:backend`) and the webpack development server (`webpack serve --mode development`).

After starting the application using any of the above methods, you can access the chatbot by navigating to:
```
http://localhost:3000/
```
(Or the port specified by your webpack dev server, typically `http://localhost:8080/` if `npm run dev` is used and the Python backend is not on port 3000).

## Project Structure

Here's an overview of the key files and directories:

```
SIT-chatbot/
├── backend/
│   ├── server.js         # Node.js backend server
│   └── server.py         # Python backend server (FastAPI)
├── src/
│   ├── app.js            # Main frontend JavaScript logic
│   ├── index.html        # Main HTML file for the frontend
│   ├── styles.css        # CSS styles for the frontend
│   └── sit-data/
│       └── logo.png      # Project logo
├── .env                  # Environment variables (create this file)
├── package.json          # npm package configuration, scripts, and frontend dependencies
├── README.md             # This file
├── requirements.txt      # Python backend dependencies
└── webpack.config.js     # Webpack configuration for building frontend assets
```

## Running via locally hosted custom LLM
If you are running the locally hosted custom LLM, you will need to create an ElevenLabs account and create an Agent. Under the LLM section, select the Custom LLM,
and enter the URL given by your ngrok console in the Server URL. Ensure the model ID is sit-chatbot.

Under the Request Headers, add a header. The type will be Value, Name will be ngrok-skip-browser-warning and set the value to true. Try to do a test call to see if it connects.



## License

This project is licensed under the [MIT License](LICENSE).