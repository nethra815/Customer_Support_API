const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");

function addMessage(text, type, metadata = null) {
  const message = document.createElement("div");
  message.className = `message ${type}-message`;

  if (type === "user") {
    message.innerHTML = `<div class="message-body"><p>${escapeHtml(text)}</p></div>`;
  } else {
    const sources = metadata?.sources?.length
      ? `<ul class="sources">${metadata.sources.map((source) => `<li>${escapeHtml(source)}</li>`).join("")}</ul>`
      : "";
    const details = metadata
      ? `<div class="metadata"><span>Relevance ${Math.round(metadata.relevance_score * 100)}%</span><span>${metadata.response_time_ms} ms</span></div>${sources}`
      : "";
    message.innerHTML = `
      <div class="avatar">V</div>
      <div class="message-body">
        <p class="message-label">VoltKart assistant</p>
        <p>${escapeHtml(text)}</p>
        ${details}
      </div>`;
  }
  conversation.appendChild(message);
  conversation.scrollTop = conversation.scrollHeight;
}

function addTypingMessage() {
  const message = document.createElement("div");
  message.className = "message assistant-message";
  message.id = "typing-message";
  message.innerHTML = '<div class="avatar">V</div><div class="message-body"><p class="message-label">VoltKart assistant</p><p class="typing">Searching the support guide...</p></div>';
  conversation.appendChild(message);
  conversation.scrollTop = conversation.scrollHeight;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[character]));
}

async function askQuestion(question) {
  addMessage(question, "user");
  addTypingMessage();
  sendButton.disabled = true;
  questionInput.disabled = true;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The support service could not answer.");
    }
    document.querySelector("#typing-message")?.remove();
    addMessage(payload.answer, "assistant", payload);
  } catch (error) {
    document.querySelector("#typing-message")?.remove();
    addMessage(error.message, "assistant");
  } finally {
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  askQuestion(question);
});

questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 110)}px`;
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.textContent));
});
