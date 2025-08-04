// --- src/app.js ---
import { Conversation } from "@elevenlabs/client";

let conversation = null;
let sttStream = null;
let isRecording = false;
let user_input = "";
let mouthAnimationInterval = null;
let currentMouthState = "M130,170 Q150,175 170,170"; // closed mouth
let currentState = "ready"; // ready, connected, recording, processing
let inputMode = "voice"; // voice or text

// Create the animated otter avatar SVG
function createAvatarSVG() {
  return `
        <svg viewBox="0 0 300 400" fill="none" xmlns="http://www.w3.org/2000/svg" class="avatar-svg otter-avatar">
            <!-- Otter body -->
            <ellipse cx="150" cy="240" rx="70" ry="90" fill="#8B5A2B" />
            
            <!-- Lighter belly -->
            <ellipse cx="150" cy="250" rx="50" ry="65" fill="#D2B48C" />
            
            <!-- Tail -->
            <path d="M95,270 C75,290 75,320 90,340 C105,360 120,350 125,340" fill="#8B5A2B" />
            
            <!-- Head -->
            <ellipse cx="150" cy="140" rx="60" ry="55" fill="#8B5A2B" />
            
            <!-- Ears -->
            <ellipse cx="105" cy="100" rx="15" ry="18" fill="#8B5A2B" />
            <ellipse cx="195" cy="100" rx="15" ry="18" fill="#8B5A2B" />
            <ellipse cx="105" cy="100" rx="8" ry="10" fill="#D2B48C" />
            <ellipse cx="195" cy="100" rx="8" ry="10" fill="#D2B48C" />
            
            <!-- Face white patch -->
            <ellipse cx="150" cy="150" rx="40" ry="35" fill="#F5F5DC" />
            
            <!-- Eyes -->
            <ellipse cx="130" cy="130" rx="10" ry="12" fill="white" />
            <ellipse cx="170" cy="130" rx="10" ry="12" fill="white" />
            <circle cx="130" cy="130" r="6" fill="#000000" />
            <circle cx="170" cy="130" r="6" fill="#000000" />
            <circle cx="128" cy="128" r="2" fill="white" />
            <circle cx="168" cy="128" r="2" fill="white" />
            
            <!-- Nose -->
           <ellipse cx="150" cy="155" rx="12" ry="8" fill="#5D4037" />
           
           <!-- Whiskers -->
           <line x1="155" y1="155" x2="190" y2="145" stroke="#5D4037" stroke-width="1.5" />
           <line x1="155" y1="158" x2="190" y2="158" stroke="#5D4037" stroke-width="1.5" />
           <line x1="155" y1="161" x2="190" y2="170" stroke="#5D4037" stroke-width="1.5" />
           <line x1="145" y1="155" x2="110" y2="145" stroke="#5D4037" stroke-width="1.5" />
           <line x1="145" y1="158" x2="110" y2="158" stroke="#5D4037" stroke-width="1.5" />
           <line x1="145" y1="161" x2="110" y2="170" stroke="#5D4037" stroke-width="1.5" />
           
           <!-- Mouth -->
           <path 
               id="avatarMouth"
               d="${currentMouthState}"
               stroke="#5D4037" 
               stroke-width="1.5" 
               fill="none"
           />
           
           <!-- SIT graduation cap -->
           <path d="M100,85 L200,85 L200,70 L100,70 Z" fill="#003B73" />
           <path d="M120,70 L180,70 L150,40 Z" fill="#003B73" />
           <path d="M150,40 L150,30 L160,25" stroke="#FFD700" stroke-width="2" />
           <circle cx="160" cy="25" r="3" fill="#FFD700" />
       </svg>
   `;
}

// Initialize avatar
function initializeAvatar() {
  const avatarWrapper = document.getElementById("animatedAvatar");
  avatarWrapper.innerHTML = createAvatarSVG();
}

// UI State Management
function updatePrimaryButton(state, text = null, icon = null) {
  const button = document.getElementById("primaryActionButton");
  const buttonText = document.getElementById("buttonText");
  const buttonIcon = document.getElementById("buttonIcon");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  currentState = state;

  // Remove all state classes
  button.classList.remove("recording", "processing");
  statusDot.classList.remove("connected", "recording", "processing");

  // Update icon
  if (icon) {
    buttonIcon.innerHTML = icon;
  }

  switch (state) {
    case "ready":
      if (!text)
        buttonText.textContent =
          inputMode === "voice" ? "Start Conversation" : "Connect";
      else buttonText.textContent = text;
      statusText.textContent = "Ready to help";
      button.disabled = false;
      buttonIcon.innerHTML = `<circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>`;
      break;

    case "connected":
      if (!text)
        buttonText.textContent =
          inputMode === "voice" ? "Start Speaking" : "Connected";
      else buttonText.textContent = text;
      statusText.textContent = "Connected - Ready to listen";
      statusDot.classList.add("connected");
      button.disabled = false;
      if (inputMode === "voice") {
        buttonIcon.innerHTML = `<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>`;
      }
      break;

    case "recording":
      buttonText.textContent = "Stop Speaking";
      statusText.textContent = "Listening...";
      button.classList.add("recording");
      statusDot.classList.add("recording");
      button.disabled = false;
      buttonIcon.innerHTML = `<rect x="6" y="6" width="12" height="12" rx="2" ry="2"></rect>`;
      break;

    case "processing":
      buttonText.textContent = "Processing...";
      statusText.textContent = "Processing your message";
      button.classList.add("processing");
      statusDot.classList.add("processing");
      button.disabled = true;
      buttonIcon.innerHTML = `<line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>`;
      break;
  }
}

function showSecondaryButton(text, action) {
  const button = document.getElementById("secondaryActionButton");
  const buttonText = document.getElementById("secondaryButtonText");

  buttonText.textContent = text;
  button.classList.remove("hidden");

  // Remove old event listeners and add new one
  const newButton = button.cloneNode(true);
  button.parentNode.replaceChild(newButton, button);
  document
    .getElementById("secondaryActionButton")
    .addEventListener("click", action);
}

function hideSecondaryButton() {
  document.getElementById("secondaryActionButton").classList.add("hidden");
}

function switchInputMode(mode) {
  inputMode = mode;
  const voiceBtn = document.getElementById("voiceModeBtn");
  const textBtn = document.getElementById("textModeBtn");
  const textContainer = document.getElementById("textInputContainer");
  const primaryButton = document.getElementById("primaryActionButton");

  if (mode === "voice") {
    voiceBtn.classList.add("active");
    textBtn.classList.remove("active");
    textContainer.classList.add("hidden");
    primaryButton.classList.remove("hidden");
    updatePrimaryButton(currentState);
  } else {
    textBtn.classList.add("active");
    voiceBtn.classList.remove("active");
    textContainer.classList.remove("hidden");
    if (currentState === "connected") {
      primaryButton.classList.add("hidden");
    }
  }
}

// Animate mouth when speaking
function startMouthAnimation() {
  if (mouthAnimationInterval) return;

  const avatarWrapper = document.getElementById("animatedAvatar");
  if (avatarWrapper) {
    avatarWrapper.classList.add("avatar-speaking");

    const speakingIndicator = document.getElementById("speakingIndicator");
    if (speakingIndicator) {
      speakingIndicator.classList.remove("hidden");
    }
  }

  mouthAnimationInterval = setInterval(() => {
    const mouthElement = document.getElementById("avatarMouth");
    if (mouthElement) {
      const shouldChangeMouth = Math.random() > 0.4;

      if (shouldChangeMouth) {
        currentMouthState =
          currentMouthState === "M130,170 Q150,175 170,170"
            ? "M130,170 Q150,195 170,170"
            : "M130,170 Q150,175 170,170";

        mouthElement.setAttribute("d", currentMouthState);
        mouthElement.setAttribute(
          "fill",
          currentMouthState.includes("195") ? "#8B4513" : "none"
        );
        mouthElement.setAttribute(
          "opacity",
          currentMouthState.includes("195") ? "0.7" : "1"
        );
      }
    }
  }, Math.random() * 200 + 100);
}

function stopMouthAnimation() {
  if (mouthAnimationInterval) {
    clearInterval(mouthAnimationInterval);
    mouthAnimationInterval = null;
  }

  const avatarWrapper = document.getElementById("animatedAvatar");
  if (avatarWrapper) {
    avatarWrapper.classList.remove("avatar-speaking");

    const speakingIndicator = document.getElementById("speakingIndicator");
    if (speakingIndicator) {
      speakingIndicator.classList.add("hidden");
    }
  }

  currentMouthState = "M130,170 Q150,175 170,170";
  const mouthElement = document.getElementById("avatarMouth");
  if (mouthElement) {
    mouthElement.setAttribute("d", currentMouthState);
    mouthElement.setAttribute("fill", "none");
    mouthElement.setAttribute("opacity", "1");
  }
}

async function requestMicrophonePermission() {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
    return true;
  } catch (error) {
    console.error("Microphone permission denied:", error);
    return false;
  }
}

async function getSignedUrl() {
  try {
    const url = "/api/signed-url";
    console.log("[Frontend] Fetching signed URL from:", url);
    const response = await fetch(url);
    console.log("[Frontend] Signed URL response status:", response.status);
    if (!response.ok) throw new Error("Failed to get signed URL");
    const data = await response.json();
    console.log("[Frontend] Signed URL data:", data);
    return data.signedUrl;
  } catch (error) {
    console.error("[Frontend] Error getting signed URL:", error);
    throw error;
  }
}

function updateStatus(isConnected) {
  const statusElement = document.getElementById("connectionStatus");
  statusElement.textContent = isConnected ? "Connected" : "Disconnected";
  statusElement.classList.toggle("connected", isConnected);
}

function addMessageToChat(message, sender) {
  const chatMessages = document.getElementById("chatMessages");
  const messageElement = document.createElement("div");
  messageElement.classList.add("message");
  if (sender === "user") {
    messageElement.classList.add("user-message");
  } else {
    messageElement.classList.add("bot-message");
  }
  messageElement.innerHTML = `
       <div class="message-content">${message}</div>
   `;
  chatMessages.appendChild(messageElement);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showError(message) {
  addMessageToChat(`❌ ${message}`, "bot");
}

function get_context(text) {
  console.log("[Frontend] Processing text through get_context:", text);
  let processedText = text.trim();
  processedText = processedText.replace(/\s+/g, " ");
  console.log("[Frontend] Text after processing:", processedText);
  return processedText;
}

async function startSpeechToText() {
  try {
    if (!conversation) {
      throw new Error(
        "Conversation not initialized. Please start the conversation first."
      );
    }

    isRecording = true;
    console.log("[Frontend] Starting STT recording");

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    sttStream = {
      mediaRecorder,
      stream,
      stop: function () {
        if (mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        }
        stream.getTracks().forEach((track) => track.stop());
      },
    };

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      try {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.wav");
        formData.append("language", "en");

        console.log("[Frontend] Sending audio to backend for transcription");
        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(
            `Server returned ${response.status}: ${response.statusText}`
          );
        }

        const result = await response.json();
        if (result && result.text) {
          console.log("[Frontend] Received transcription:", result.text);
          user_input = result.text;

          if (isRecording) {
            addMessageToChat(`💭 "${result.text}"`, "user");
            updatePrimaryButton("connected");
            showSecondaryButton("Send Message", handleSendToAgent);
          }
        }
      } catch (error) {
        console.error("[Frontend] Error getting transcription:", error);
        showError("Failed to transcribe speech: " + error.message);
        updatePrimaryButton("connected");
      }
    });

    mediaRecorder.start();
    return sttStream;
  } catch (error) {
    console.error("[Frontend] Error starting speech to text:", error);
    showError("Failed to start speech recognition. " + error.message);
    isRecording = false;
    updatePrimaryButton("connected");
    return null;
  }
}

async function stopSpeechToText() {
  if (isRecording) {
    try {
      if (sttStream && typeof sttStream.stop === "function") {
        sttStream.stop();
        console.log("[Frontend] MediaRecorder stopped");
        isRecording = false;
        return "";
      }
    } catch (error) {
      console.error("[Frontend] Error stopping STT stream:", error);
      isRecording = false;
    }
  }
  return "";
}

async function sendProcessedText(text) {
  if (!text) return;

  const processedText = get_context(text);
  console.log(
    "[Frontend] Sending processed text to conversation:",
    processedText
  );

  if (!conversation) {
    console.error("Conversation not initialized");
    showError(
      "Not connected to the agent. Please start the conversation first."
    );
    return;
  }

  startMouthAnimation();

  // Debug: Log available methods on conversation object
  console.log(
    "🔍 [DEBUG] Conversation object methods:",
    Object.getOwnPropertyNames(conversation)
  );
  console.log(
    "🔍 [DEBUG] Conversation prototype methods:",
    Object.getOwnPropertyNames(Object.getPrototypeOf(conversation))
  );
  console.log(
    "🔍 [DEBUG] Available methods:",
    Object.getOwnPropertyNames(conversation).filter(
      (prop) => typeof conversation[prop] === "function"
    )
  );

  try {
    if (typeof conversation.sendUserMessage === "function") {
      await conversation.sendUserMessage(processedText);
      console.log("Message sent using sendUserMessage");
    } else {
      console.error("No suitable method found to send text to conversation");
      console.error(
        "🔍 [DEBUG] conversation.sendUserMessage type:",
        typeof conversation.sendUserMessage
      );
      showError("Could not send message - no suitable method found.");
    }
  } catch (error) {
    console.error("[Frontend] Error sending processed text:", error);
    showError("Failed to send your message.");
  }
}

async function sendTextMessage(text) {
  if (!text.trim()) return;

  addMessageToChat(text, "user");
  updatePrimaryButton("processing");
  startMouthAnimation();

  try {
    await sendProcessedText(text);
  } catch (error) {
    console.error("[Frontend] Error sending text message:", error);
    showError("Failed to send your message.");
    stopMouthAnimation();
    updatePrimaryButton("connected");
  }
}

async function startConversation() {
  try {
    updatePrimaryButton("processing", "Connecting...");

    const hasPermission = await requestMicrophonePermission();
    if (!hasPermission) {
      showError("Microphone permission is required for voice features.");
      updatePrimaryButton("ready");
      return;
    }

    const signedUrl = await getSignedUrl();

    conversation = await Conversation.startSession({
      signedUrl,
      disableTts: false,
      disableSpeechToText: true,
      disableAudioProcessing: true,
      disableAutostart: true,
      voiceEnabled: false,
      autostart: false,
      microphoneEnabled: false,
      inputMode: "text",
      speechToTextEnabled: false,
      onConnect: () => {
        console.log("🔗 [CONNECTION] Connected to conversation");
        updateStatus(true);
        updatePrimaryButton("connected");

        setTimeout(() => {
          try {
            if (
              conversation &&
              typeof conversation.endVoiceSession === "function"
            ) {
              conversation.endVoiceSession();
            }
            if (
              conversation &&
              typeof conversation.setMicrophoneState === "function"
            ) {
              conversation.setMicrophoneState(false);
            }
          } catch (e) {
            console.log("Error stopping voice activity:", e);
          }
        }, 100);
      },
      onDisconnect: () => {
        console.log("🔗 [CONNECTION] Disconnected from conversation");
        updateStatus(false);
        updatePrimaryButton("ready");
        hideSecondaryButton();
        stopMouthAnimation();
      },
      onError: (error) => {
        console.error("🚨 [ERROR] Conversation error:", error);
        updatePrimaryButton("ready");
        hideSecondaryButton();
        showError("An error occurred during the conversation.");
      },
      onMessage: (msg) => {
        // Enhanced logging for all message types
        console.log(
          "📨 [MESSAGE RECEIVED] Raw message:",
          JSON.stringify(msg, null, 2)
        );
        console.log("📨 [MESSAGE KEYS]:", Object.keys(msg));

        let botResponse = null;
        let messageType = "unknown";
        let messageSource = msg.source || "unknown";

        // Check for bot utterance
        if (msg.type === "bot_utterance" && msg.text) {
          botResponse = msg.text;
          messageType = "bot_utterance";
          console.log("🤖 [BOT_UTTERANCE]:", botResponse);
        }
        // Check for TTS response
        else if (msg.type === "tts" && msg.text) {
          botResponse = msg.text;
          messageType = "tts";
          console.log("🤖 [TTS]:", botResponse);
        }
        // Check for AI source message
        else if (msg.source === "ai" && msg.message) {
          botResponse = msg.message;
          messageType = "ai_source";
          console.log("🤖 [AI_SOURCE]:", botResponse);
        }
        // Check for response type message
        else if (msg.type === "response" && msg.response) {
          botResponse = msg.response;
          messageType = "response";
          console.log("🤖 [RESPONSE]:", botResponse);
        }
        // Check for completion type message
        else if (msg.type === "completion" && msg.text) {
          botResponse = msg.text;
          messageType = "completion";
          console.log("🤖 [COMPLETION]:", botResponse);
        }
        // Check for message content
        else if (msg.content && typeof msg.content === "string") {
          botResponse = msg.content;
          messageType = "content";
          console.log("🤖 [CONTENT]:", botResponse);
        }
        // Check for data field
        else if (msg.data && typeof msg.data === "string") {
          botResponse = msg.data;
          messageType = "data";
          console.log("🤖 [DATA]:", botResponse);
        }
        // Fallback for any text field (excluding user transcripts)
        else if (
          msg.text &&
          msg.type !== "user_transcript" &&
          msg.type !== "transcript"
        ) {
          botResponse = msg.text;
          messageType = "fallback_text";
          console.log("🤖 [FALLBACK_TEXT]:", botResponse);
        }

        // Handle user transcripts separately
        if (msg.type === "user_transcript" || msg.type === "transcript") {
          console.log(
            "👤 [USER_TRANSCRIPT]:",
            msg.text || msg.content || "No text found"
          );
          return;
        }

        // Handle bot responses
        if (botResponse) {
          // Comprehensive chatbot response logging
          console.log("🤖 [CHATBOT_RESPONSE_DETECTED]:", {
            timestamp: new Date().toISOString(),
            messageType: messageType,
            source: messageSource,
            response: botResponse,
            responseLength: botResponse.length,
            wordCount: botResponse.split(" ").length,
            originalMessage: msg,
          });

          // Additional separate logs for visibility
          console.log("=".repeat(50));
          console.log("🤖 CHATBOT SAID:", botResponse);
          console.log("=".repeat(50));

          addMessageToChat(botResponse, "bot");
          stopMouthAnimation();
          updatePrimaryButton("connected");
        } else {
          console.log("❓ [UNHANDLED_MESSAGE]:", {
            type: msg.type,
            source: msg.source,
            keys: Object.keys(msg),
            fullMessage: msg,
          });
        }
      },
      onModeChange: (mode) => {
        console.log("🎵 [MODE_CHANGE]:", mode);
        const isSpeaking = mode && mode.mode === "speaking";
        if (isSpeaking) {
          console.log("🗣️ [SPEAKING] Bot started speaking");
          startMouthAnimation();
        } else {
          console.log("🤫 [NOT_SPEAKING] Bot stopped speaking");
          stopMouthAnimation();
        }
      },
    });
  } catch (error) {
    console.error("🚨 [ERROR] Error starting conversation:", error);
    updatePrimaryButton("ready");
    showError("Failed to start conversation. Please try again.");
  }
}

async function endConversation() {
  try {
    if (isRecording) {
      await stopSpeechToText();
    }

    if (conversation) {
      await conversation.endSession();
      conversation = null;
      sttStream = null;
    }

    user_input = "";
    isRecording = false;

    updatePrimaryButton("ready");
    hideSecondaryButton();
    updateStatus(false);
    stopMouthAnimation();
  } catch (error) {
    console.error("Error ending conversation:", error);
    showError(
      "Failed to properly end conversation. You may need to refresh the page."
    );
  }
}

async function handlePrimaryAction() {
  switch (currentState) {
    case "ready":
      await startConversation();
      break;
    case "connected":
      if (inputMode === "voice") {
        updatePrimaryButton("recording");
        await startSpeechToText();
      }
      break;
    case "recording":
      updatePrimaryButton("processing");
      await stopSpeechToText();
      break;
  }
}

async function handleSendToAgent() {
  if (!user_input) return;

  hideSecondaryButton();
  updatePrimaryButton("processing");

  try {
    await sendProcessedText(user_input);
    user_input = "";
    updatePrimaryButton("connected");
  } catch (error) {
    console.error("Error sending to agent:", error);
    updatePrimaryButton("connected");
  }
}

// Initialize when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", async () => {
  console.log("🚀 [INIT] Application starting...");
  updateStatus(false);
  initializeAvatar();
  updatePrimaryButton("ready");

  // Primary action button
  document
    .getElementById("primaryActionButton")
    .addEventListener("click", handlePrimaryAction);

  // Input mode toggle
  document
    .getElementById("voiceModeBtn")
    .addEventListener("click", () => switchInputMode("voice"));
  document
    .getElementById("textModeBtn")
    .addEventListener("click", () => switchInputMode("text"));

  // Text input handling
  const textInput = document.getElementById("textInput");
  const sendTextButton = document.getElementById("sendTextButton");

  sendTextButton.addEventListener("click", () => {
    const text = textInput.value.trim();
    if (text) {
      console.log("📝 [USER_INPUT] Text message:", text);
      sendTextMessage(text);
      textInput.value = "";
    }
  });

  textInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      const text = textInput.value.trim();
      if (text) {
        console.log("📝 [USER_INPUT] Text message (Enter):", text);
        sendTextMessage(text);
        textInput.value = "";
      }
    }
  });

  // Initialize with voice mode
  switchInputMode("voice");
  console.log("✅ [INIT] Application initialized successfully");
});
