const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const messages = document.getElementById("chat-messages");

function addMessage(text, sender, sources) {
    const div = document.createElement("div");
    div.className = `message ${sender}`;
    div.textContent = text;

    if (sources && sources.length > 0) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = `Sources: ${sources.join(", ")}`;
        div.appendChild(src);
    }

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return div;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    input.value = "";
    input.disabled = true;

    const loadingMsg = addMessage("Thinking...", "bot");

    try {
        const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();

        loadingMsg.remove();

        if (data.error) {
            addMessage(`Error: ${data.error}`, "bot error");
        } else {
            addMessage(data.answer, "bot", data.sources);
        }
    } catch (err) {
        loadingMsg.remove();
        addMessage(`Network error: ${err.message}`, "bot error");
    } finally {
        input.disabled = false;
        input.focus();
    }
});
