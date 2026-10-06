// ========================================================
// Automaton AI - Standalone Floating Chatbot Controller
// Interacts with FastAPI Zero-Hallucination RAG Backend
// ========================================================

const API_BASE_URL = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? window.location.origin
  : "http://127.0.0.1:8000";

let conversationHistory = [];
let isWindowOpen = true; // Open by default for immediate interaction

document.addEventListener("DOMContentLoaded", () => {
  const launcher = document.getElementById("chatLauncherBtn");
  const windowEl = document.getElementById("floatingChatbotWindow");

  // Keep launcher hidden when window is open
  if (isWindowOpen) {
    launcher.classList.add("hidden");
    windowEl.classList.remove("closed");
  } else {
    launcher.classList.remove("hidden");
    windowEl.classList.add("closed");
  }

  // Focus input automatically
  const inputEl = document.getElementById("chatInputField");
  if (inputEl) inputEl.focus();
});

// Toggle Chatbot Window Open / Minimized
function toggleChatWindow() {
  const launcher = document.getElementById("chatLauncherBtn");
  const windowEl = document.getElementById("floatingChatbotWindow");

  isWindowOpen = !isWindowOpen;

  if (isWindowOpen) {
    windowEl.classList.remove("closed");
    launcher.classList.add("hidden");
    const inputEl = document.getElementById("chatInputField");
    if (inputEl) inputEl.focus();
  } else {
    windowEl.classList.add("closed");
    launcher.classList.remove("hidden");
  }
}

// Handle Prompt Chips
function handlePromptChip(text) {
  sendMessage(text);
}

// Handle Form Submission
function handleChatSubmit(event) {
  event.preventDefault();
  const inputEl = document.getElementById("chatInputField");
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text);
}

// Send Message to Backend RAG Pipeline
async function sendMessage(queryText) {
  renderUserMessage(queryText);
  const typingId = renderTypingIndicator();

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: queryText,
        history: conversationHistory.slice(-4)
      })
    });

    removeTypingIndicator(typingId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    conversationHistory.push({ role: "user", content: queryText });
    conversationHistory.push({ role: "assistant", content: data.response });

    renderAssistantMessage(data);

  } catch (error) {
    removeTypingIndicator(typingId);
    renderErrorMessage(`Connection error: ${error.message}. Please verify FastAPI is running on port 8000.`);
  }

  scrollStream();
}

function renderUserMessage(text) {
  const stream = document.getElementById("chatStreamBody");
  const html = `
    <div class="message-row user">
      <div class="message-bubble user">${escapeHtml(text)}</div>
    </div>
  `;
  stream.insertAdjacentHTML("beforeend", html);
  scrollStream();
}

function renderAssistantMessage(data) {
  const stream = document.getElementById("chatStreamBody");
  const formattedContent = formatMarkdown(data.response);

  // Citations
  let citationsHtml = "";
  if (data.citations && data.citations.length > 0) {
    citationsHtml = `
      <div class="citation-badges-box">
        ${data.citations.map(c => `
          <a href="${c.url || 'https://automatonai.com'}" target="_blank" class="citation-pill" title="${escapeHtml(c.snippet)}">
            <span>🔗</span> ${escapeHtml(c.title)}
          </a>
        `).join("")}
      </div>
    `;
  }

  // Follow-up chips
  let followupsHtml = "";
  if (data.suggested_followups && data.suggested_followups.length > 0) {
    followupsHtml = `
      <div class="prompt-chips-wrap" style="margin-top: 10px;">
        ${data.suggested_followups.map(f => `
          <button class="prompt-chip" onclick="handlePromptChip('${escapeHtml(f)}')">
            ⚡ ${escapeHtml(f)}
          </button>
        `).join("")}
      </div>
    `;
  }

  // Attribution note
  const attributionHtml = `
    <div class="attribution-sub">
      <span style="color: var(--orange-accent);">●</span>
      <span>Confidence: ${(data.confidence * 100).toFixed(1)}% • Grounded in Automaton AI Documentation (Pune, India)</span>
    </div>
  `;

  const html = `
    <div class="message-row assistant">
      <div class="message-bubble assistant">
        ${formattedContent}
        ${citationsHtml}
        ${attributionHtml}
        ${followupsHtml}
      </div>
    </div>
  `;

  stream.insertAdjacentHTML("beforeend", html);
  scrollStream();
}

function renderTypingIndicator() {
  const stream = document.getElementById("chatStreamBody");
  const id = `typing-${Date.now()}`;
  const html = `
    <div class="message-row assistant" id="${id}">
      <div class="message-bubble assistant" style="color: var(--slate-muted); font-style: italic; font-size: 12px;">
        <span>🔍 Verifying knowledge base & retrieving citations...</span>
      </div>
    </div>
  `;
  stream.insertAdjacentHTML("beforeend", html);
  scrollStream();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderErrorMessage(msg) {
  const stream = document.getElementById("chatStreamBody");
  const html = `
    <div class="message-row assistant">
      <div class="message-bubble assistant" style="border-color: #EF4444; color: #FCA5A5;">
        <strong>Error</strong><br>
        ${escapeHtml(msg)}
      </div>
    </div>
  `;
  stream.insertAdjacentHTML("beforeend", html);
  scrollStream();
}

function scrollStream() {
  const stream = document.getElementById("chatStreamBody");
  if (stream) {
    stream.scrollTop = stream.scrollHeight;
  }
}

function clearActiveChat() {
  conversationHistory = [];
  const defaultHtml = `
    <div class="chat-welcome-card">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
        <strong style="color: var(--cyan-electric); font-size: 14px;">Welcome to Automaton AI</strong>
        <span style="font-size: 10px; color: var(--orange-accent); background: rgba(255,107,53,0.15); padding: 2px 6px; border-radius: 4px; font-weight: 600;">SLA Grounded</span>
      </div>
      <p style="color: var(--slate-muted); font-size: 12px; line-height: 1.5;">
        I am the official enterprise virtual assistant for <strong>Automaton AI Infosystem</strong> (Hinjewadi, Pune). 
        Ask me about ADVIT Studio V2, Model Gallery, Data Zoo datasets, or scheduling an enterprise demo.
      </p>
      <div class="prompt-chips-wrap">
        <button class="prompt-chip" onclick="handlePromptChip('What is ADVIT Studio V2?')">⚡ ADVIT Studio V2</button>
        <button class="prompt-chip" onclick="handlePromptChip('How does managed data labeling work?')">🏷️ Managed Labeling</button>
        <button class="prompt-chip" onclick="handlePromptChip('What datasets are in Data Zoo?')">🦁 Data Zoo Datasets</button>
        <button class="prompt-chip" onclick="handlePromptChip('How much does ADVIT Studio cost?')">💰 Explore Pricing</button>
        <button class="prompt-chip" onclick="handlePromptChip('Tell me about the Apollo Tyres case study')">🏭 Apollo Tyres</button>
        <button class="prompt-chip" onclick="handlePromptChip('Where is Automaton AI headquartered?')">📍 Pune Office</button>
      </div>
    </div>
  `;
  document.getElementById("chatStreamBody").innerHTML = defaultHtml;
}

// Markdown formatting helper
function formatMarkdown(text) {
  if (!text) return "";
  let out = escapeHtml(text);

  // Bold
  out = out.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Headings
  out = out.replace(/^### (.*$)/gim, "<h4>$1</h4>");
  out = out.replace(/^## (.*$)/gim, "<h4>$1</h4>");

  // Bullets
  out = out.replace(/^\- (.*$)/gim, "<li>$1</li>");
  out = out.replace(/^\* (.*$)/gim, "<li>$1</li>");
  out = out.replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>");

  // Blockquotes / note
  out = out.replace(/^> (.*$)/gim, "<div style='border-left: 2px solid var(--cyan-electric); padding-left: 8px; color: var(--slate-muted); margin: 6px 0;'>$1</div>");

  // Links
  out = out.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" style="color: var(--cyan-electric); text-decoration: underline;">$1</a>');

  // Newlines
  out = out.replace(/\n\n/g, "<br><br>");
  out = out.replace(/\n/g, "<br>");

  return out;
}

function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
