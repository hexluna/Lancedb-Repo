\
# SIT Chatbot Codebase Documentation

## 1. `backend/server.js`

This file sets up a Node.js server using the Express framework. Its primary responsibilities include:

*   **Environment Configuration**: Loads environment variables using `dotenv`.
*   **Middleware**:
    *   `cors`: Enables Cross-Origin Resource Sharing, allowing requests from different origins.
    *   `express.json`: Parses incoming requests with JSON payloads.
    *   **Static File Serving**: Serves static files (like CSS, JavaScript bundles, images) from the `../dist` directory under the `/static` route.
    *   **Request Logging**: A custom middleware logs the method and URL of every incoming request along with a timestamp.
*   **API Endpoints**:
    *   `GET /api/signed-url`: Asynchronously fetches a signed URL from the ElevenLabs API for the conversational AI agent. It retrieves `AGENT_ID` and `XI_API_KEY` from environment variables. It includes error handling and logs various stages of the request.
    *   `GET /api/getAgentId`: Returns the `AGENT_ID` from the environment variables. This is likely used by the frontend to identify the agent.
*   **Fallback Route**:
    *   `GET *`: A catch-all route that serves the `index.html` file from the `../dist` directory. This is common in Single Page Applications (SPAs) to ensure all routes are handled by the frontend router after the initial load.
*   **Server Initialization**: Starts the server, listening on the port defined by the `PORT` environment variable, or `3000` if not specified. It logs the server's running URL to the console.

## 2. `backend/server.py`

This file implements a Python backend server using the FastAPI framework. Its key functions are:

*   **Environment Configuration**: Loads environment variables using `dotenv`.
*   **FastAPI Application**: Initializes a FastAPI application instance.
*   **CORS Middleware**: Configures Cross-Origin Resource Sharing to allow requests from all origins, with all methods and headers, and allows credentials.
*   **API Endpoints**:
    *   `GET /api/signed-url`: Asynchronously fetches a signed URL from the ElevenLabs API. It uses the `AGENT_ID` (specifically noted as "SIT otter assistant agent ID") and `XI_API_KEY` from environment variables. It uses `httpx` for making the asynchronous HTTP request and includes error handling, raising an `HTTPException` if environment variables are missing or if the API call fails.
    *   `GET /api/getAgentId`: Returns the `AGENT_ID` from environment variables.
*   **Static File Serving**:
    *   `app.mount("/static", StaticFiles(directory="dist"), name="static")`: Mounts a directory named `dist` to serve static files under the `/static` path.
*   **Root Route**:
    *   `GET /`: Serves the `index.html` file from the `dist` directory when the root path is accessed.
*   **Server Execution (Implicit)**: While not explicitly shown in this snippet, FastAPI applications are typically run using an ASGI server like Uvicorn.

## 3. `src/index.html`

This is the main HTML file for the SIT Chatbot application. Its structure and purpose are:

*   **Metadata**:
    *   `charset`, `viewport`: Standard meta tags for character encoding and responsive viewport configuration.
    *   `title`: "SIT Chatbot".
    *   `description`: "Chat with SIT's friendly otter assistant".
    *   `icon`: Sets the favicon for the page.
*   **Styling and Fonts**:
    *   Links to `/static/styles.css` for the application's styling.
    *   Preconnects to Google Fonts and imports the "Inter" font family.
*   **Body Layout**:
    *   `<div class="chat-container">`: The main container for the chat interface.
    *   `<div id="connectionStatus" class="hidden">Disconnected</div>`: Displays the connection status to the backend/service (initially hidden).
    *   `<div id="speakingStatus" class="hidden">Agent Silent</div>`: Displays whether the AI agent is currently speaking (initially hidden).
    *   `<div class="chat-interface">`: A grid layout containing the two main panels:
        *   **Left Panel (`<div class="chat-card">`)**:
            *   **Header**: Displays the SIT logo and title ("SIT Chatbot", "Your friendly SIT assistant").
            *   **Chat Messages (`<div class="chat-messages" id="chatMessages">`)**: This area is where conversation messages (user and bot) will be dynamically inserted by JavaScript.
            *   **Chat Input Area (`<div class="chat-input-area">`)**:
                *   `<button id="startButton">`: "Start Conversation" button with a play icon.
                *   `<button id="endButton">`: "Stop Conversation" button with a stop/cross icon, initially disabled and hidden.
        *   **Right Panel (`<div class="avatar-card">`)**:
            *   `<div id="animatedAvatar" class="avatar-wrapper">`: This container is where the animated otter avatar (SVG) will be inserted by JavaScript.
*   **JavaScript Inclusion**:
    *   `<script src="/static/bundle.js"></script>`: Includes the bundled JavaScript file, which presumably contains the application logic from `app.js` and any other client-side scripts after being processed by a tool like Webpack (implied by `bundle.js` and the presence of `webpack.config.js` in the workspace).

## 4. `src/styles.css`

This CSS file defines the visual appearance and layout of the SIT Chatbot interface. Key aspects include:

*   **Global Styles**:
    *   `body`: Sets a default `font-family` (Inter, with fallbacks), a linear gradient background using SIT blue colors, and an animated particle background effect (`body::before` with `@keyframes float`).
*   **Layout Containers**:
    *   `.chat-container`: Centers the chat interface on the page and manages its overall dimensions.
    *   `.chat-interface`: Defines a two-column grid layout for the chat card and the avatar card. The grid proportions are adjusted to give more space to the chat.
*   **Status Indicators**:
    *   `#connectionStatus`, `#speakingStatus`: Styles for the fixed-position status indicators (top-right corner), including background colors for different states (connected/disconnected, speaking/silent).
*   **Cards**:
    *   `.chat-card`, `.avatar-card`: Styles for the main content cards, including background (semi-transparent white with backdrop blur), border-radius, box-shadow, padding, and sizing. Hover effects are defined for `.control-card` (likely a shared class or an intended class for these cards).
*   **Header**:
    *   `.card-header`, `.header-logo`, `.sit-logo`, `.header-text`: Styles for the header section within the chat card, including logo display and text formatting.
*   **Chat Area**:
    *   `.chat-messages`: Styles for the area displaying chat messages.
    *   `.message`, `.message-content`, `.user-message`, `.bot-message`: Styles for individual messages, differentiating between user and bot messages.
    *   `.chat-input-area`, `.chat-input`, `.send-button`, `.voice-button`: Styles for the input field and buttons (though the input field and send button are not present in the provided `index.html`, these styles suggest they might be part of a more complete version or a planned feature).
*   **Recording/Control Buttons**:
    *   `.record-button`, `.stop-button`: Styles for the start/stop conversation buttons, including hover, active, and disabled states.
*   **Avatar**:
    *   `.avatar-wrapper`, `.avatar-svg`, `.otter-avatar`: Styles for the avatar container and the SVG element itself.
    *   `.avatar-speaking`: A class likely added via JavaScript to trigger speaking animations.
    *   `.mouth-closed`, `.mouth-open`, `@keyframes mouthClosed`, `@keyframes mouthOpen`: Styles and animations for the otter's mouth movement (though the actual SVG manipulation for mouth state is handled in `app.js`).
    *   `.avatar-speaking::after`, `@keyframes pulse`: A pulsing animation effect when the avatar is speaking.
*   **Responsive Design**:
    *   `@media (max-width: 768px)` and `@media (max-width: 600px)`: Media queries to adjust layout and styles for smaller screens (tablets and mobile phones).
*   **Animations & Effects**:
    *   Subtle slide-in animations for `.control-group` elements.
    *   Loading spinner animation for `.record-button.loading`.
*   **Fallback/Other Avatar Styles**:
    *   Includes styles for `.doctor-avatar`, `.celebrity-avatar-container`, suggesting flexibility for different avatar types, though the current implementation focuses on the otter.

## 5. `src/app.js`

This JavaScript file contains the client-side logic for the SIT Chatbot. It manages the user interface, avatar animation, and communication with the backend for the conversation.

*   **Global Variables**:
    *   `conversation`: Stores the conversation object/state.
    *   `mouthAnimationInterval`: Holds the interval ID for the mouth animation.
    *   `currentMouthState`: Stores the current SVG path data for the otter's mouth, defaulting to a closed state.
*   **Avatar Management**:
    *   `createAvatarSVG()`: Generates the SVG string for the animated otter avatar. The SVG includes elements for the body, head, facial features, and a graduation cap. The mouth path (`<path id="avatarMouth">`) is dynamically set using `currentMouthState`.
    *   `initializeAvatar()`: Inserts the generated SVG into the `#animatedAvatar` div in the HTML.
*   **Mouth Animation**:
    *   `startMouthAnimation()`: Intended to animate the otter's mouth when the agent is speaking. It sets an interval to change the mouth state. The provided snippet is incomplete (`if (mouthAnimationInterval) ` and `setInterval(() => {…}, ...)` have empty or placeholder logic).
    *   `stopMouthAnimation()`: Intended to stop the mouth animation and reset the mouth to a closed state. The provided snippet is incomplete.
*   **Permissions and API Calls**:
    *   `requestMicrophonePermission()`: Asynchronously requests microphone permission from the user (implementation details are omitted with `{…}`).
    *   `getSignedUrl()`: Asynchronously fetches the signed URL from the backend API (`/api/signed-url`) (implementation details are omitted with `{…}`). This URL is likely used to establish the WebSocket connection for the conversation.
*   **UI Updates**:
    *   `updateStatus(isConnected)`: Updates the `#connectionStatus` element's text and class based on the connection state.
    *   `updateSpeakingStatus(mode)`: Updates the `#speakingStatus` element based on whether the agent is speaking. It also logs this status to the console.
    *   `setFormControlsState(disabled)`: Enables or disables form controls (though `userInput` and `sendButton` are referenced, they are not in the provided `index.html`).
*   **Chat Interface Setup**:
    *   `setupChatInterface()`: Initializes the avatar. Notes that there's no text input or send button to wire up in the current HTML structure.
*   **Conversation Flow (stubbed functions)**:
    *   `sendMessage()`: Placeholder for sending a message.
    *   `addMessageToChat(message, sender)`: Placeholder for adding a message to the chat display.
    *   `setInputEnabled(enabled)`: Placeholder for enabling/disabling input.
    *   `showError(message)`: Placeholder for displaying an error message.
    *   `initializeConversation()`: Placeholder for initializing the conversation.
    *   `startConversation()`: Placeholder for starting the conversation.
    *   `endConversation()`: Placeholder for ending the conversation.
*   **Initialization**:
    *   `document.addEventListener('DOMContentLoaded', async () => {…})`: An event listener that runs when the HTML document is fully loaded. The specific actions within this async function are omitted. It's the main entry point for the frontend application logic.
