/**
 * InsureAI Concierge — Frontend App (Alpine.js)
 *
 * Features:
 *   - Multi-turn chat with session memory
 *   - Voice input (Web Speech API)
 *   - File upload (PDF / image) with text extraction
 *   - Claim status tracker
 *   - Analytics dashboard
 *   - Agent detail inspector (JSON modal)
 *   - Fraud risk badges
 *   - Tool call display
 *   - Language detection indicator
 */

function app() {
  return {
    // ── State ─────────────────────────────────────────────────────────────────
    activeTab: 'chat',
    messages: [],
    inputText: '',
    isLoading: false,
    sessionId: generateSessionId(),
    sessionTokens: 0,
    totalCalls: 0,
    lastLatency: null,
    lastLang: null,
    healthOk: false,
    claimsFiledCount: 0,

    // Analytics
    analytics: null,
    abResults: {},
    claims: [],
    claimLookupNum: '',
    lookedUpClaim: null,

    // Policies for sidebar
    policies: [],

    // Notifications
    notifications: [],
    unreadCount: 0,

    // Voice
    isRecording: false,
    recognition: null,

    // File upload
    uploadedDoc: null,
    isDragging: false,

    // Modal
    modal: { open: false, title: '', content: '' },

    // Quick starters
    starters: [
      'I was in a car accident yesterday, damage about $5,000',
      'My house had water damage from a burst pipe',
      'I need to file a claim for my stolen car',
      'Does my policy cover rental cars?',
      'What is my deductible for POL-AUTO-001?',
      'Check status of my claim CLM-2026-',
    ],

    // ── Init ──────────────────────────────────────────────────────────────────
    async init() {
      await this.checkHealth();
      await this.loadPolicies();
      await this.loadTokenStats();
      this.setupDragDrop();
      this.setupVoice();
      // Auto-refresh token stats every 30s
      setInterval(() => this.loadTokenStats(), 30000);
    },

    // ── Health check ──────────────────────────────────────────────────────────
    async checkHealth() {
      try {
        const r = await fetch('/api/health');
        this.healthOk = r.ok;
      } catch {
        this.healthOk = false;
      }
    },

    // ── Load policies for sidebar ─────────────────────────────────────────────
    async loadPolicies() {
      try {
        const r = await fetch('/api/policies');
        if (r.ok) this.policies = await r.json();
      } catch {}
    },

    // ── Token stats ───────────────────────────────────────────────────────────
    async loadTokenStats() {
      try {
        const r = await fetch('/api/stats');
        if (r.ok) {
          const d = await r.json();
          this.sessionTokens = d.total_tokens || 0;
          this.totalCalls = d.total_calls || 0;
        }
      } catch {}
    },

    // ── New session ───────────────────────────────────────────────────────────
    newSession() {
      if (!confirm('Start a new session? Current conversation will be cleared from view.')) return;
      this.sessionId = generateSessionId();
      this.messages = [];
      this.uploadedDoc = null;
      this.claimsFiledCount = 0;
      this.lastLang = null;
    },

    // ── Send message ──────────────────────────────────────────────────────────
    async sendMessage(overrideText) {
      const text = (overrideText || this.inputText).trim();
      if (!text || this.isLoading) return;

      this.inputText = '';
      this.resetTextarea();

      const userMsg = {
        role: 'user',
        content: text,
        time: nowStr(),
      };
      this.messages.push(userMsg);
      this.scrollToBottom();

      this.isLoading = true;

      try {
        const payload = {
          message: text,
          session_id: this.sessionId,
        };
        if (this.uploadedDoc) {
          payload.document_context = this.uploadedDoc.extracted_text;
          this.uploadedDoc = null;
        }

        const r = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }

        const data = await r.json();
        this.handleResponse(data);
      } catch (e) {
        this.messages.push({
          role: 'assistant',
          content: `⚠ Error: ${e.message}. Please try again.`,
          time: nowStr(),
        });
      } finally {
        this.isLoading = false;
        this.scrollToBottom();
        await this.loadTokenStats();
      }
    },

    // ── Handle API response ───────────────────────────────────────────────────
    handleResponse(data) {
      this.lastLatency = data.latency_ms;
      this.lastLang = data.language_detected;

      // Build agent details for inspector
      const details = {};
      if (data.guardrail_details) details['Agent 1: Guardrail'] = data.guardrail_details;
      if (data.sentiment_details) details['Agent 0.5: Sentiment'] = data.sentiment_details;
      if (data.claim_parser_details) details['Agent 2: Claim Parser'] = data.claim_parser_details;
      if (data.safety_details) details['Agent 3: Safety Check'] = data.safety_details;
      if (data.fraud_details) details['Agent 4: Fraud Detector'] = data.fraud_details;
      if (data.tool_calls && data.tool_calls.length > 0) details['Tools Used'] = data.tool_calls;
      if (data.rag_context && data.rag_context.length > 0) details['RAG Context'] = data.rag_context;

      // Build metadata pills
      const meta = {};
      if (data.intent) meta.intent = data.intent;
      if (data.priority) meta.priority = data.priority;
      if (data.estimated_loss_amount) meta.loss = data.estimated_loss_amount;
      if (data.claim_number) {
        meta.claim_number = data.claim_number;
        this.claimsFiledCount++;
      }
      if (data.fraud_details?.fraud_risk_score) meta.fraud_risk = data.fraud_details.fraud_risk_score;
      if (data.escalation_ticket_id) meta.escalation_ticket = data.escalation_ticket_id;
      if (data.tool_calls && data.tool_calls.length > 0) meta.tools = data.tool_calls.map(t => t.tool);
      if (data.language_detected && data.language_detected !== 'en') meta.language = data.language_detected;
      if (data.sentiment_details?.sentiment && data.sentiment_details.sentiment !== 'calm') {
        meta.sentiment = data.sentiment_details.sentiment;
      }
      if (data.settlement_range) meta.settlement = data.settlement_range;

      // Token usage for this call
      if (data.token_usage) {
        this.sessionTokens = (this.sessionTokens || 0) + (data.token_usage.total_tokens || 0);
      }

      // Check for notifications
      if (data.claim_number) {
        this.loadNotifications();
      }

      const msg = {
        role: 'assistant',
        content: data.final_response || 'No response generated.',
        time: nowStr(),
        meta: Object.keys(meta).length > 0 ? meta : null,
        details: Object.keys(details).length > 0 ? details : null,
        showDetails: false,
        followUps: data.follow_up_questions || [],
        checklist: data.evidence_checklist || [],
        settlement: data.settlement_range || null,
        sessionId: data.session_id,
      };

      this.messages.push(msg);
    },

    // ── Starter messages ──────────────────────────────────────────────────────
    sendStarter(text) {
      this.inputText = text;
      this.sendMessage();
    },

    // ── Voice input ───────────────────────────────────────────────────────────
    setupVoice() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) return;

      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = 'en-US';

      this.recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        this.inputText = transcript;
      };

      this.recognition.onend = () => {
        this.isRecording = false;
      };

      this.recognition.onerror = (e) => {
        this.isRecording = false;
        if (e.error !== 'aborted') {
          console.warn('Speech recognition error:', e.error);
        }
      };
    },

    toggleVoice() {
      if (!this.recognition) {
        alert('Voice input not supported in this browser. Try Chrome.');
        return;
      }
      if (this.isRecording) {
        this.recognition.stop();
        this.isRecording = false;
      } else {
        this.recognition.start();
        this.isRecording = true;
      }
    },

    // ── File upload ───────────────────────────────────────────────────────────
    async uploadFile(event) {
      const file = event.target.files?.[0];
      if (!file) return;
      await this.processUpload(file);
      event.target.value = '';
    },

    setupDragDrop() {
      const chatEl = document.getElementById('chatMessages');
      if (!chatEl) return;
      chatEl.addEventListener('dragover', (e) => { e.preventDefault(); this.isDragging = true; });
      chatEl.addEventListener('dragleave', () => { this.isDragging = false; });
      chatEl.addEventListener('drop', (e) => { e.preventDefault(); this.isDragging = false; this.handleDrop(e); });
    },

    async handleDrop(event) {
      const file = event.dataTransfer?.files?.[0];
      if (file) await this.processUpload(file);
    },

    async processUpload(file) {
      const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/webp', 'image/gif'];
      const maxSize = 10 * 1024 * 1024;
      if (file.size > maxSize) { alert('File too large. Max 10MB.'); return; }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', this.sessionId);

      // Show uploading state
      this.messages.push({
        role: 'assistant',
        content: `📎 Uploading ${file.name}...`,
        time: nowStr(),
      });
      this.scrollToBottom();

      try {
        const r = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!r.ok) throw new Error(`Upload failed: ${r.statusText}`);
        const data = await r.json();

        this.uploadedDoc = {
          filename: data.filename,
          size_bytes: data.size_bytes,
          extracted_text: data.extracted_text,
          preview: data.preview,
        };

        // Replace the uploading message
        this.messages[this.messages.length - 1] = {
          role: 'assistant',
          content: `📎 Document ready: **${data.filename}**\n\nPreview: "${data.preview?.slice(0, 150) || 'No text extracted'}..."\n\nThe document content will be included in your next message. Type your question about it.`,
          time: nowStr(),
        };
      } catch (e) {
        this.messages[this.messages.length - 1] = {
          role: 'assistant',
          content: `⚠ Upload failed: ${e.message}`,
          time: nowStr(),
        };
      }
      this.scrollToBottom();
    },

    // ── Notifications ─────────────────────────────────────────────────────────
    async loadNotifications() {
      try {
        const r = await fetch(`/api/notifications?session_id=${this.sessionId}`);
        if (r.ok) {
          const d = await r.json();
          this.notifications = d.notifications || [];
          this.unreadCount = this.notifications.filter(n => !n.is_read).length;
        }
      } catch {}
    },

    async markNotificationsRead() {
      try {
        await fetch(`/api/notifications/read?session_id=${this.sessionId}`, { method: 'POST' });
        this.unreadCount = 0;
        this.notifications.forEach(n => n.is_read = 1);
      } catch {}
    },

    // ── Export session ────────────────────────────────────────────────────────
    exportSession() {
      window.open(`/api/session/${this.sessionId}/export`, '_blank');
    },

    // ── Claims tracker ────────────────────────────────────────────────────────
    async loadClaims() {
      try {
        const r = await fetch('/api/claims?limit=30');
        if (r.ok) this.claims = await r.json();
      } catch {}
    },

    async lookupClaim() {
      const num = this.claimLookupNum.trim();
      if (!num) return;
      try {
        const r = await fetch(`/api/claim/${encodeURIComponent(num)}`);
        if (r.ok) {
          this.lookedUpClaim = await r.json();
        } else {
          this.lookedUpClaim = null;
          alert(`Claim "${num}" not found.`);
        }
      } catch (e) {
        alert('Error looking up claim: ' + e.message);
      }
    },

    statusColor(status) {
      const map = {
        'Submitted': 'bg-blue-900/50 text-blue-300',
        'Under Review': 'bg-yellow-900/50 text-yellow-300',
        'Adjuster Assigned': 'bg-purple-900/50 text-purple-300',
        'Settled': 'bg-green-900/50 text-green-300',
        'Closed': 'bg-gray-700 text-gray-400',
        'open': 'bg-red-900/50 text-red-300',
      };
      return map[status] || 'bg-gray-700 text-gray-400';
    },

    // ── Analytics ─────────────────────────────────────────────────────────────
    async loadAnalytics() {
      try {
        const [r1, r2] = await Promise.all([
          fetch('/api/analytics'),
          fetch('/api/ab-results'),
        ]);
        if (r1.ok) this.analytics = await r1.json();
        if (r2.ok) this.abResults = await r2.json();
      } catch {}
    },

    // ── Modal ─────────────────────────────────────────────────────────────────
    openModal(title, data) {
      this.modal.title = title;
      this.modal.content = JSON.stringify(data, null, 2);
      this.modal.open = true;
    },

    copyModal() {
      navigator.clipboard.writeText(this.modal.content)
        .then(() => alert('Copied to clipboard!'))
        .catch(() => {});
    },

    // ── Scroll helpers ────────────────────────────────────────────────────────
    scrollToBottom() {
      this.$nextTick(() => {
        const el = document.getElementById('chatMessages');
        if (el) el.scrollTop = el.scrollHeight;
      });
    },

    autoResize(event) {
      const el = event.target;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    },

    resetTextarea() {
      this.$nextTick(() => {
        const el = document.getElementById('msgInput');
        if (el) { el.style.height = 'auto'; el.style.height = '44px'; }
      });
    },
  };
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function generateSessionId() {
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function nowStr() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
