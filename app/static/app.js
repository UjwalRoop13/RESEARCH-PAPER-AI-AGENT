(function () {
  "use strict";

  const state = {
    sessionId: localStorage.getItem("paperpilot_session_id") || null,
    papers: [],
    selectedPaperIds: new Set(),
  };

  const el = (id) => document.getElementById(id);
  const escapeHtml = (str) =>
    String(str ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: opts.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const j = await res.json();
        detail = j.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const contentType = res.headers.get("content-type") || "";
    return contentType.includes("application/json") ? res.json() : res.text();
  }

  // ---------------- health ----------------
  async function checkHealth() {
    try {
      const h = await api("/api/health");
      el("modeIndicator").textContent = h.mock_llm ? "MOCK MODE (no API key)" : "LIVE \u00b7 " + h.embedding_backend + " embeddings";
      el("statusText").textContent = h.mock_llm ? "mock mode" : "online";
      el("statusDot").style.background = h.mock_llm ? "var(--amber)" : "var(--teal)";
    } catch (e) {
      el("statusText").textContent = "offline";
      el("statusDot").style.background = "var(--magenta)";
    }
  }

  // ---------------- tabs ----------------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
      el("view-" + btn.dataset.tab).classList.remove("hidden");
      el("chatInputBar").style.display = btn.dataset.tab === "chat" ? "flex" : "none";
      if (btn.dataset.tab === "notes") loadNotes();
      if (btn.dataset.tab === "reports") loadReports();
    });
  });

  // ---------------- upload ----------------
  el("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const statusEl = el("uploadStatus");
    statusEl.innerHTML = '<span class="radar"></span> Uploading & ingesting...';
    const form = new FormData();
    form.append("file", file);
    try {
      const paper = await api("/api/papers/upload", { method: "POST", body: form });
      statusEl.textContent = "Ingested: " + paper.title;
      await loadPapers();
    } catch (err) {
      statusEl.textContent = "Upload failed: " + err.message;
    } finally {
      e.target.value = "";
      setTimeout(() => (statusEl.textContent = ""), 6000);
    }
  });

  // ---------------- paper list ----------------
  async function loadPapers() {
    try {
      state.papers = await api("/api/papers");
    } catch (e) {
      state.papers = [];
    }
    renderPapers();
  }

  function renderPapers() {
    const list = el("paperList");
    if (!state.papers.length) {
      list.innerHTML = '<div class="empty-mini">No papers uploaded yet. Upload a PDF to begin, or use the Search tab to discover papers on the web.</div>';
      return;
    }
    list.innerHTML = state.papers
      .map((p) => {
        const selected = state.selectedPaperIds.has(p.paper_id) ? "selected" : "";
        const authors = (p.authors || []).join(", ") || "Unknown authors";
        return `
        <div class="paper-card ${selected}" data-id="${p.paper_id}">
          <div class="pt">${escapeHtml(p.title)}</div>
          <div class="pm"><span class="status-dot status-${p.status}"></span>${escapeHtml(authors)} \u00b7 ${p.page_count ?? "?"}p \u00b7 ${p.status}</div>
          <button class="del" data-id="${p.paper_id}" title="Delete">&times;</button>
        </div>`;
      })
      .join("");

    list.querySelectorAll(".paper-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.classList.contains("del")) return;
        const id = card.dataset.id;
        if (state.selectedPaperIds.has(id)) state.selectedPaperIds.delete(id);
        else state.selectedPaperIds.add(id);
        renderPapers();
      });
    });
    list.querySelectorAll(".del").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm("Delete this paper and all its chunks/notes?")) return;
        await api("/api/papers/" + btn.dataset.id, { method: "DELETE" });
        state.selectedPaperIds.delete(btn.dataset.id);
        await loadPapers();
      });
    });
  }

  // ---------------- chat ----------------
  function appendMessage(role, content, citations = [], toolTrace = []) {
    const scroll = el("chatScroll");
    const hint = scroll.querySelector(".hint");
    if (hint) hint.remove();

    const div = document.createElement("div");
    div.className = "msg " + role;
    const traceHtml = toolTrace.length
      ? '<div class="tool-trace">' +
        toolTrace.map((t) => `<span class="tool-chip ${t.status}">${escapeHtml(t.tool_name)} \u00b7 ${t.latency_ms}ms</span>`).join("") +
        "</div>"
      : "";
    const citeHtml = citations.length
      ? '<div class="citations">' + citations.map((c) => `<span>${escapeHtml(c.paper_id).slice(0, 8)}\u2026 p.${c.page_number}</span>`).join("") + "</div>"
      : "";
    div.innerHTML = `
      <div class="who">${role === "user" ? "You" : "PaperPilot Agent"}</div>
      <div class="bubble">${escapeHtml(content)}</div>
      ${traceHtml}${citeHtml}
    `;
    scroll.appendChild(div);
    scroll.scrollIntoView({ block: "end" });
    window.scrollTo(0, document.body.scrollHeight);
    div.scrollIntoView({ block: "end" });
  }

  async function sendChat() {
    const input = el("chatInput");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    appendMessage("user", text);
    const btn = el("chatSendBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="radar"></span>';

    try {
      const result = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, session_id: state.sessionId }),
      });
      state.sessionId = result.session_id;
      localStorage.setItem("paperpilot_session_id", state.sessionId);
      appendMessage("assistant", result.content, result.citations, result.tool_trace);
    } catch (err) {
      appendMessage("assistant", "Error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Send";
    }
  }

  el("chatSendBtn").addEventListener("click", sendChat);
  el("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  // ---------------- search ----------------
  async function doSearch() {
    const q = el("searchInput").value.trim();
    if (!q) return;
    const resultsEl = el("searchResults");
    resultsEl.innerHTML = '<div class="hint"><span class="radar"></span> Searching the web...</div>';
    try {
      const results = await api("/api/papers/search?q=" + encodeURIComponent(q));
      if (!results.length) {
        resultsEl.innerHTML = '<div class="hint">No results found. Try a different topic.</div>';
        return;
      }
      resultsEl.innerHTML = results
        .map(
          (r) => `
        <div class="search-result">
          <div class="st">${escapeHtml(r.title)}</div>
          <div class="sm">${escapeHtml(r.authors || "")} \u00b7 ${escapeHtml(r.year || "")} \u00b7 ${escapeHtml(r.venue || "")}</div>
          <div class="ss">${escapeHtml(r.summary || "")}</div>
          ${r.url ? `<div style="margin-top:8px;"><a href="${encodeURI(r.url)}" target="_blank" rel="noopener" style="font-family:var(--mono);font-size:11px;">Open source &#8599;</a></div>` : ""}
        </div>`
        )
        .join("");
    } catch (err) {
      resultsEl.innerHTML = '<div class="hint">Search failed: ' + escapeHtml(err.message) + "</div>";
    }
  }
  el("searchBtn").addEventListener("click", doSearch);
  el("searchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
  });

  // ---------------- notes ----------------
  async function loadNotes() {
    const listEl = el("notesList");
    listEl.innerHTML = '<div class="hint">Loading notes...</div>';
    try {
      const notes = await api("/api/notes");
      if (!notes.length) {
        listEl.innerHTML = '<div class="hint">No notes yet. Ask the agent to "save a note" during chat, or it will save one automatically when you ask it to remember something.</div>';
        return;
      }
      listEl.innerHTML = notes
        .map(
          (n) => `
        <div class="note-card">
          <div class="meta">${n.paper_id ? "linked to paper " + escapeHtml(n.paper_id).slice(0, 8) + "\u2026" : "unlinked"} \u00b7 ${new Date(n.created_at * 1000).toLocaleString()}</div>
          <p>${escapeHtml(n.content)}</p>
        </div>`
        )
        .join("");
    } catch (e) {
      listEl.innerHTML = '<div class="hint">Failed to load notes.</div>';
    }
  }

  // ---------------- reports ----------------
  async function loadReports() {
    const listEl = el("reportsList");
    listEl.innerHTML = '<div class="hint">Loading reports...</div>';
    try {
      const reports = await api("/api/reports");
      if (!reports.length) {
        listEl.innerHTML = '<div class="hint">No reports yet. Ask the agent in chat to "write a literature review" or "generate a report" across selected papers.</div>';
        return;
      }
      listEl.innerHTML = reports
        .map(
          (r) => `
        <div class="report-card">
          <h3>${escapeHtml(r.title)}</h3>
          <div class="meta">${escapeHtml(r.report_type)} \u00b7 ${(r.paper_ids || []).length} paper(s) \u00b7 ${new Date(r.created_at * 1000).toLocaleString()}</div>
          <div class="export-row">
            <a href="/api/reports/${r.report_id}/export?format=md">Export .md</a>
            <a href="/api/reports/${r.report_id}/export?format=docx">Export .docx</a>
            <a href="/api/reports/${r.report_id}/export?format=pdf">Export .pdf</a>
          </div>
        </div>`
        )
        .join("");
    } catch (e) {
      listEl.innerHTML = '<div class="hint">Failed to load reports.</div>';
    }
  }

  // ---------------- init ----------------
  checkHealth();
  loadPapers();
  setInterval(loadPapers, 8000); // cheap polling so ingestion status updates without manual refresh
})();
