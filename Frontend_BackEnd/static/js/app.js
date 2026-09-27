// TelcoPulse CRM — Enterprise Customer Retention & Intelligence Platform Logic
const API_BASE = "";

// State
let currentAccountData = null;
let currentPrediction = null;
let currentDirectoryRecords = [];
let currentOutreachData = {
  email: "",
  call: "",
  sms: ""
};
let activeChannel = "email";
let crmPageNumber = 1;
let crmPageLimit = 10;

// Preconfigured Realistic Customer Personas
const CUSTOMER_PERSONAS = {
  high_risk: {
    full_name: "Eleanor Vance",
    email: "eleanor.vance@demo-telecom.invalid",
    phone: "+1 (555) ***-9182",
    initials: "EV",
    avatar_color: "#3b82f6",
    assigned_agent: "Alex Rivera",
    sentiment: "At-Risk / Price Sensitive",
    sentiment_class: "at-risk",
    last_touchpoint: "3 days ago • Inbound Support Call",
    profile: {
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "No",
      Dependents: "No",
      tenure: 2,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: 89.95,
      TotalCharges: 179.90
    }
  },
  loyal_power: {
    full_name: "Marcus Brody",
    email: "marcus.brody@demo-telecom.invalid",
    phone: "+1 (555) ***-4412",
    initials: "MB",
    avatar_color: "#8b5cf6",
    assigned_agent: "Jordan Reed",
    sentiment: "High Loyalty / Satisfied",
    sentiment_class: "satisfied",
    last_touchpoint: "2 weeks ago • Annual Account Review",
    profile: {
      gender: "Male",
      SeniorCitizen: 0,
      Partner: "Yes",
      Dependents: "Yes",
      tenure: 64,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "Fiber optic",
      OnlineSecurity: "Yes",
      OnlineBackup: "Yes",
      DeviceProtection: "Yes",
      TechSupport: "Yes",
      StreamingTV: "Yes",
      StreamingMovies: "Yes",
      Contract: "Two year",
      PaperlessBilling: "Yes",
      PaymentMethod: "Bank transfer (automatic)",
      MonthlyCharges: 104.50,
      TotalCharges: 6688.00
    }
  },
  budget_user: {
    full_name: "Sarah Jenkins",
    email: "sarah.jenkins@demo-telecom.invalid",
    phone: "+1 (555) ***-3105",
    initials: "SJ",
    avatar_color: "#10b981",
    assigned_agent: "Taylor Morgan",
    sentiment: "Neutral / Cost Sensitive",
    sentiment_class: "neutral",
    last_touchpoint: "1 month ago • Autopay Notification",
    profile: {
      gender: "Female",
      SeniorCitizen: 0,
      Partner: "No",
      Dependents: "No",
      tenure: 18,
      PhoneService: "Yes",
      MultipleLines: "No",
      InternetService: "DSL",
      OnlineSecurity: "Yes",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: "One year",
      PaperlessBilling: "No",
      PaymentMethod: "Mailed check",
      MonthlyCharges: 34.20,
      TotalCharges: 615.60
    }
  },
  tech_unprotected: {
    full_name: "David Miller",
    email: "david.miller@demo-telecom.invalid",
    phone: "+1 (555) ***-8023",
    initials: "DM",
    avatar_color: "#f59e0b",
    assigned_agent: "Alex Rivera",
    sentiment: "Vulnerable / Lacks Support",
    sentiment_class: "at-risk",
    last_touchpoint: "5 days ago • Billing Inquiry Call",
    profile: {
      gender: "Male",
      SeniorCitizen: 1,
      Partner: "No",
      Dependents: "No",
      tenure: 5,
      PhoneService: "Yes",
      MultipleLines: "Yes",
      InternetService: "Fiber optic",
      OnlineSecurity: "No",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "Yes",
      Contract: "Month-to-month",
      PaperlessBilling: "Yes",
      PaymentMethod: "Electronic check",
      MonthlyCharges: 82.30,
      TotalCharges: 411.50
    }
  }
};

// Theme Management (Default: Light Mode)
function initTheme() {
  const savedTheme = localStorage.getItem("crm-theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme") || "light";
  const newTheme = current === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", newTheme);
  localStorage.setItem("crm-theme", newTheme);
  updateThemeButton(newTheme);
}

function updateThemeButton(theme) {
  const icon = document.getElementById("theme-toggle-icon");
  const text = document.getElementById("theme-toggle-text");
  if (icon && text) {
    if (theme === "light") {
      icon.textContent = "☀️";
      text.textContent = "Light Mode";
    } else {
      icon.textContent = "🌙";
      text.textContent = "Dark Mode";
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initFormBindings();
  initEventListeners();
  initAuth();
  fetchSystemMetrics();
  loadSubscriberDirectory();

  // Load default persona
  loadCustomerPersona("high_risk");
});

function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add("active");
    });
  });
}

function initFormBindings() {
  const tenureInput = document.getElementById("input-tenure");
  const tenureVal = document.getElementById("val-tenure");
  if (tenureInput && tenureVal) {
    tenureInput.addEventListener("input", (e) => {
      tenureVal.textContent = `${e.target.value} mo`;
      autoUpdateTotalCharges();
    });
  }

  const monthlyInput = document.getElementById("input-monthly");
  const monthlyVal = document.getElementById("val-monthly");
  if (monthlyInput && monthlyVal) {
    monthlyInput.addEventListener("input", (e) => {
      monthlyVal.textContent = `$${parseFloat(e.target.value).toFixed(2)}`;
      autoUpdateTotalCharges();
    });
  }

  const form = document.getElementById("subscriber-form");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      assessCustomerHealth();
    });
  }
}

// Authentication Helpers & Modal Management
function getAuthHeaders(extra = {}) {
  const token = sessionStorage.getItem("telcopulse_auth_token") || "";
  const headers = { ...extra };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
    headers["X-API-Key"] = token;
  }
  return headers;
}

function showAuthModal(errMsg = "") {
  const modal = document.getElementById("auth-modal");
  const errEl = document.getElementById("auth-error-msg");
  const input = document.getElementById("auth-token-input");
  if (errEl) {
    errEl.textContent = errMsg;
    errEl.style.display = errMsg ? "block" : "none";
  }
  if (input) {
    input.value = sessionStorage.getItem("telcopulse_auth_token") || "";
  }
  if (modal) modal.style.display = "flex";
}

function hideAuthModal() {
  const modal = document.getElementById("auth-modal");
  if (modal) modal.style.display = "none";
}

async function updateAuthBadge() {
  const badgeText = document.getElementById("auth-status-text");
  const badgeIcon = document.getElementById("auth-status-icon");
  try {
    const res = await fetch(`${API_BASE}/api/auth/status`, {
      headers: getAuthHeaders()
    });
    if (res.ok) {
      const data = await res.json();
      if (!data.auth_required) {
        if (badgeText) badgeText.textContent = "Auth: Open (Dev)";
        if (badgeIcon) badgeIcon.textContent = "🔓";
      } else if (data.authenticated) {
        if (badgeText) badgeText.textContent = "Auth: Verified";
        if (badgeIcon) badgeIcon.textContent = "🔒";
      } else {
        if (badgeText) badgeText.textContent = "Auth: Required";
        if (badgeIcon) badgeIcon.textContent = "🔑";
      }
    }
  } catch (err) {
    // Keep default badge text if network error
  }
}

function initAuth() {
  updateAuthBadge();
  document.getElementById("btn-auth-status")?.addEventListener("click", () => {
    showAuthModal();
  });
  document.getElementById("auth-modal-close")?.addEventListener("click", hideAuthModal);
  document.getElementById("auth-clear-btn")?.addEventListener("click", () => {
    sessionStorage.removeItem("telcopulse_auth_token");
    const input = document.getElementById("auth-token-input");
    if (input) input.value = "";
    hideAuthModal();
    updateAuthBadge();
  });
  document.getElementById("auth-save-btn")?.addEventListener("click", async () => {
    const input = document.getElementById("auth-token-input");
    const val = input ? input.value.trim() : "";
    if (val) {
      sessionStorage.setItem("telcopulse_auth_token", val);
    } else {
      sessionStorage.removeItem("telcopulse_auth_token");
    }
    hideAuthModal();
    updateAuthBadge();
  });
}

function initEventListeners() {
  // Theme Toggle
  document.getElementById("theme-toggle")?.addEventListener("click", toggleTheme);

  // Quick Persona Chips
  document.querySelectorAll(".quick-customer-chip[data-persona]").forEach(chip => {
    chip.addEventListener("click", () => {
      loadCustomerPersona(chip.getAttribute("data-persona"));
    });
  });

  // Load Demo JSON button
  document.getElementById("btn-load-demo-json")?.addEventListener("click", loadDemoJsonFile);

  // Upload Single Customer JSON button & file input
  const singleJsonInput = document.getElementById("single-json-input");
  document.getElementById("btn-upload-customer-json")?.addEventListener("click", () => {
    singleJsonInput?.click();
  });
  singleJsonInput?.addEventListener("change", (e) => {
    handleSingleJsonUpload(e.target);
  });

  // Channel switch buttons
  document.querySelectorAll(".channel-btn[data-channel]").forEach(btn => {
    btn.addEventListener("click", () => {
      switchChannel(btn.getAttribute("data-channel"));
    });
  });

  // Copy Channel Content button
  document.getElementById("btn-copy-channel")?.addEventListener("click", copyActiveChannelContent);

  // Directory Filters & Search
  document.getElementById("dir-search-input")?.addEventListener("input", () => {
    crmPageNumber = 1;
    loadSubscriberDirectory();
  });
  document.getElementById("dir-filter-contract")?.addEventListener("change", () => {
    crmPageNumber = 1;
    loadSubscriberDirectory();
  });
  document.getElementById("dir-filter-status")?.addEventListener("change", () => {
    crmPageNumber = 1;
    loadSubscriberDirectory();
  });

  // Directory Pagination
  document.getElementById("btn-dir-prev")?.addEventListener("click", prevDirectoryPage);
  document.getElementById("btn-dir-next")?.addEventListener("click", nextDirectoryPage);

  // Directory Row Event Delegation for "Open Account"
  document.getElementById("directory-tbody")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-open-cust-account");
    if (btn) {
      const idx = parseInt(btn.getAttribute("data-index"), 10);
      if (!isNaN(idx) && currentDirectoryRecords[idx]) {
        openCustomerAccount(currentDirectoryRecords[idx]);
      }
    }
  });

  // Recommendation 1-Click Action Delegation
  document.getElementById("action-items-container")?.addEventListener("click", (e) => {
    const btn = e.target.closest(".btn-apply-action");
    if (btn) applyIntervention(btn);
  });

  // Batch Upload File Input and Buttons
  const batchFileInput = document.getElementById("batch-file-input");
  document.getElementById("btn-trigger-batch-upload")?.addEventListener("click", () => {
    batchFileInput?.click();
  });
  batchFileInput?.addEventListener("change", (e) => {
    handleBatchFileSelect(e.target);
  });
  document.getElementById("btn-load-demo-batch")?.addEventListener("click", loadDemoBatchJson);
}

function autoUpdateTotalCharges() {
  const tenure = parseFloat(document.getElementById("input-tenure")?.value) || 1;
  const monthly = parseFloat(document.getElementById("input-monthly")?.value) || 50;
  const totalInput = document.getElementById("input-total");
  if (totalInput) {
    totalInput.value = (tenure * monthly).toFixed(2);
  }
}

// Safe Form Helpers
const setVal = (id, val) => {
  const el = document.getElementById(id);
  if (el) el.value = val;
};
const getVal = (id, fallback = "") => {
  const el = document.getElementById(id);
  return el ? el.value : fallback;
};

function loadCustomerPersona(key) {
  const persona = CUSTOMER_PERSONAS[key];
  if (!persona) return;

  currentAccountData = persona;

  // Update UI Card Identity
  const nameEl = document.getElementById("cust-display-name");
  const emailEl = document.getElementById("cust-display-email");
  const phoneEl = document.getElementById("cust-display-phone");
  const avatarBox = document.getElementById("cust-avatar-box");
  const sentimentEl = document.getElementById("cust-display-sentiment");
  const touchEl = document.getElementById("cust-display-touch");

  if (nameEl) nameEl.textContent = persona.full_name;
  if (emailEl) emailEl.textContent = persona.email;
  if (phoneEl) phoneEl.textContent = persona.phone;
  if (avatarBox) {
    avatarBox.textContent = persona.initials;
    avatarBox.style.backgroundColor = persona.avatar_color;
  }
  if (sentimentEl) {
    sentimentEl.textContent = persona.sentiment;
    sentimentEl.className = `sentiment-badge ${persona.sentiment_class}`;
  }
  if (touchEl) touchEl.textContent = persona.last_touchpoint;

  // Fill Form Controls
  const p = persona.profile;
  setVal("input-gender", p.gender);
  setVal("input-senior", p.SeniorCitizen);
  setVal("input-partner", p.Partner);
  setVal("input-dependents", p.Dependents);

  const tenureEl = document.getElementById("input-tenure");
  if (tenureEl) tenureEl.value = p.tenure;
  const valTenure = document.getElementById("val-tenure");
  if (valTenure) valTenure.textContent = `${p.tenure} mo`;

  setVal("input-phoneservice", p.PhoneService);
  setVal("input-multiplelines", p.MultipleLines);
  setVal("input-internetservice", p.InternetService);
  setVal("input-security", p.OnlineSecurity);
  setVal("input-backup", p.OnlineBackup);
  setVal("input-protection", p.DeviceProtection);
  setVal("input-techsupport", p.TechSupport);
  setVal("input-streamingtv", p.StreamingTV || "No");
  setVal("input-streamingmovies", p.StreamingMovies || "No");
  setVal("input-contract", p.Contract);
  setVal("input-paperless", p.PaperlessBilling);
  setVal("input-payment", p.PaymentMethod);

  const monthlyEl = document.getElementById("input-monthly");
  if (monthlyEl) monthlyEl.value = p.MonthlyCharges;
  const valMonthly = document.getElementById("val-monthly");
  if (valMonthly) valMonthly.textContent = `$${p.MonthlyCharges.toFixed(2)}`;

  setVal("input-total", p.TotalCharges);

  assessCustomerHealth();
}

function getFormPayload() {
  return {
    gender: getVal("input-gender", "Female"),
    SeniorCitizen: parseInt(getVal("input-senior", "0")) || 0,
    Partner: getVal("input-partner", "No"),
    Dependents: getVal("input-dependents", "No"),
    tenure: parseFloat(getVal("input-tenure", "1")) || 1,
    PhoneService: getVal("input-phoneservice", "Yes"),
    MultipleLines: getVal("input-multiplelines", "No"),
    InternetService: getVal("input-internetservice", "Fiber optic"),
    OnlineSecurity: getVal("input-security", "No"),
    OnlineBackup: getVal("input-backup", "No"),
    DeviceProtection: getVal("input-protection", "No"),
    TechSupport: getVal("input-techsupport", "No"),
    StreamingTV: getVal("input-streamingtv", "No"),
    StreamingMovies: getVal("input-streamingmovies", "No"),
    Contract: getVal("input-contract", "Month-to-month"),
    PaperlessBilling: getVal("input-paperless", "Yes"),
    PaymentMethod: getVal("input-payment", "Electronic check"),
    MonthlyCharges: parseFloat(getVal("input-monthly", "50")) || 50,
    TotalCharges: parseFloat(getVal("input-total", "50")) || 50,
    selected_model: "XGBoost"
  };
}

async function assessCustomerHealth() {
  const payload = getFormPayload();
  const btn = document.getElementById("btn-assess-account");
  if (btn) btn.innerHTML = "<span>⚡ Aria Analyzing Account Health...</span>";

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: getAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload)
    });

    if (res.status === 401 || res.status === 403) {
      showAuthModal("Authentication required to evaluate account health.");
      throw new Error("Authentication required (401/403)");
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      throw new Error(errData?.detail || errData?.error || "Inference API error");
    }
    const data = await res.json();
    currentPrediction = data;
    renderCopilotAssessment(data);

    const evalStatus = document.getElementById("copilot-eval-status");
    if (evalStatus) evalStatus.innerHTML = "● Analysis Live";

    // Also trigger humanized outreach generation in background
    generateOutreachDrafts(payload, data);
  } catch (err) {
    console.error("Assessment failed:", err);
    const evalStatus = document.getElementById("copilot-eval-status");
    if (evalStatus) evalStatus.innerHTML = `<span style="color:var(--rose);">⚠️ ${err.message}</span>`;
  } finally {
    if (btn) btn.innerHTML = "<span>⚡ Evaluate Account Retention Health</span>";
  }
}

function renderCopilotAssessment(data) {
  const healthScore = data.health_score || Math.max(5, Math.round((1.0 - data.churn_probability) * 100));
  const churnPct = Math.round(data.churn_probability * 100);

  // Health Dial
  const healthNum = document.getElementById("account-health-num");
  if (healthNum) {
    healthNum.textContent = healthScore;
    healthNum.style.color = data.risk_color;
  }

  const dialBox = document.getElementById("account-dial-box");
  if (dialBox) {
    dialBox.style.borderColor = data.risk_color;
    dialBox.style.boxShadow = `0 4px 14px ${data.risk_color}33`;
  }

  // Churn Prob Text
  const churnText = document.getElementById("churn-prob-text");
  if (churnText) churnText.textContent = `${churnPct}% churn probability`;

  // Status Tag
  const statusTag = document.getElementById("account-status-tag");
  if (statusTag) {
    statusTag.className = "risk-level-tag";
    if (churnPct >= 70) {
      statusTag.textContent = "CRITICAL CHURN RISK (FLIGHT IMMINENT)";
      statusTag.classList.add("critical");
    } else if (churnPct >= 50) {
      statusTag.textContent = "ELEVATED CHURN RISK";
      statusTag.classList.add("high");
    } else if (churnPct >= 30) {
      statusTag.textContent = "MODERATE RISK";
      statusTag.classList.add("moderate");
    } else {
      statusTag.textContent = "HEALTHY ACCOUNT (LOYAL SUBSCRIBER)";
      statusTag.classList.add("healthy");
    }
  }

  // Persona Tag
  const personaTag = document.getElementById("account-persona-tag");
  if (personaTag && data.cluster_info) {
    personaTag.innerHTML = `🧬 <strong>Persona ${data.cluster_id}:</strong> ${data.cluster_info.name}`;
  }

  // Plain-English Why at Risk (Drivers)
  const driversList = document.getElementById("account-drivers-list");
  if (driversList && data.factor_insights) {
    driversList.innerHTML = data.factor_insights.map(f => `
      <li class="talking-point-item">
        <strong>${f.factor}:</strong>&nbsp;${f.detail}&nbsp;<span style="color:${f.type === 'risk' ? 'var(--rose)' : 'var(--emerald)'}; font-weight:700;">(${f.impact})</span>
      </li>
    `).join("");
  }

  // Agent Talking Points
  const talkingPointsList = document.getElementById("agent-talking-points");
  if (talkingPointsList) {
    const points = data.talking_points || [
      "Acknowledge their account tenure and loyalty before discussing rates.",
      "Highlight our guaranteed 1-year rate freeze to eliminate unexpected bills.",
      "Offer complimentary 24/7 VIP dedicated tech support."
    ];
    talkingPointsList.innerHTML = points.map(pt => `
      <li class="talking-point-item">${pt}</li>
    `).join("");
  }

  // 1-Click Action Interventions
  const actionItemsContainer = document.getElementById("action-items-container");
  if (actionItemsContainer && data.recommendations) {
    actionItemsContainer.innerHTML = data.recommendations.map((rec, idx) => `
      <div class="action-card-item">
        <span class="action-card-text">${rec}</span>
        <button class="btn-apply-action" id="action-btn-${idx}" data-action-idx="${idx}">
          Apply Now ⚡
        </button>
      </div>
    `).join("");
  }
}

function applyIntervention(btn) {
  btn.textContent = "Applied ✓";
  btn.classList.add("applied");
}

async function generateOutreachDrafts(payload, predictData) {
  const custName = currentAccountData?.full_name || "Valued Subscriber";
  const prob = predictData?.churn_probability || 0.65;
  const offer = prob >= 0.5 ? "20% Annual Contract Discount & 3-Mo VIP Tech Support" : "Complimentary Speed Boost Upgrade";

  try {
    const res = await fetch(`${API_BASE}/api/generate-email`, {
      method: "POST",
      headers: getAuthHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        customer_name: custName,
        customer_data: payload,
        churn_probability: prob,
        retention_offer: offer,
        tone: "Empathetic & Personalized"
      })
    });

    if (res.ok) {
      const data = await res.json();
      currentOutreachData.email = `Subject: ${data.subject}\n\n${data.body}`;
      currentOutreachData.call = data.call_script || predictData.phone_script || "Call Script Loading...";
      currentOutreachData.sms = data.sms_body || `Hi ${custName}, we'd love to apply a special loyalty discount to your TelcoPulse account. Reply YES to confirm!`;
      updateChannelDisplay();
    }
  } catch (err) {
    console.error("Outreach generation failed:", err);
  }
}

function switchChannel(channel) {
  activeChannel = channel;
  document.querySelectorAll(".channel-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = Array.from(document.querySelectorAll(".channel-btn")).find(btn => btn.textContent.toLowerCase().includes(channel));
  if (activeBtn) activeBtn.classList.add("active");
  updateChannelDisplay();
}

function updateChannelDisplay() {
  const box = document.getElementById("copilot-channel-content");
  if (!box) return;
  box.textContent = currentOutreachData[activeChannel] || "Generating customized outreach message...";
}

function copyActiveChannelContent() {
  const box = document.getElementById("copilot-channel-content");
  if (box && box.textContent) {
    navigator.clipboard.writeText(box.textContent);
    const copyBtn = document.getElementById("btn-copy-channel");
    if (copyBtn) {
      copyBtn.textContent = "Copied! ✓";
      setTimeout(() => copyBtn.textContent = "Copy 📋", 2000);
    }
  }
}

// Fetch System KPIs & Segments
async function fetchSystemMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/analytics`);
    const data = await res.json();

    if (data.summary) {
      document.getElementById("kpi-total-accounts").textContent = data.summary.total_customers.toLocaleString();
      document.getElementById("kpi-health-rate").textContent = `${(100 - data.summary.churn_rate_pct).toFixed(1)}%`;
      document.getElementById("kpi-rev-saved").textContent = `$${((data.summary.churned_customers * data.summary.avg_monthly_charges * 12) / 1000000).toFixed(2)}M`;
      document.getElementById("kpi-avg-bill").textContent = `$${data.summary.avg_monthly_charges.toFixed(2)}`;
    }

    renderPersonasGrid(data.cluster_profiles);
  } catch (err) {
    console.error("Failed to fetch system analytics:", err);
  }
}

function renderPersonasGrid(profiles) {
  const container = document.getElementById("personas-grid-container");
  if (!container || !profiles) return;

  container.innerHTML = Object.entries(profiles).map(([id, info]) => `
    <div class="persona-segment-card">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 0.65rem;">
          <h3 style="font-size: 1.05rem; font-weight:700; color: var(--text-main);">Persona ${id}: ${info.name}</h3>
          <span class="enterprise-tag">Segment ${id}</span>
        </div>
        <p style="font-size: 0.83rem; color: var(--text-muted); line-height: 1.45;">${info.description}</p>
        
        <div class="segment-metrics-strip">
          <div>
            <div class="seg-val">${info.avg_tenure}</div>
            <div class="seg-lbl">Avg Tenure</div>
          </div>
          <div>
            <div class="seg-val">${info.avg_monthly}</div>
            <div class="seg-lbl">Monthly ARPU</div>
          </div>
          <div>
            <div class="seg-val" style="color: var(--rose);">${info.churn_propensity}</div>
            <div class="seg-lbl">Churn Propensity</div>
          </div>
        </div>
      </div>

      <div style="background: var(--bg-subtle); padding: 0.85rem; border-radius: 8px; border: 1px solid var(--border-color); font-size: 0.8rem;">
        <div style="font-weight: 600; color: var(--primary); margin-bottom: 0.2rem;">💡 Retention Action Playbook:</div>
        <div style="color: var(--text-main); line-height: 1.4;">${info.recommended_strategy}</div>
      </div>
    </div>
  `).join("");
}

// Subscribers CRM Directory
async function loadSubscriberDirectory() {
  const search = document.getElementById("dir-search-input")?.value || "";
  const contract = document.getElementById("dir-filter-contract")?.value || "All";
  const status = document.getElementById("dir-filter-status")?.value || "All";

  try {
    const params = new URLSearchParams({
      page: crmPageNumber,
      limit: crmPageLimit,
      search: search,
      contract: contract,
      churn: status
    });

    const res = await fetch(`${API_BASE}/api/customers?${params.toString()}`, {
      headers: getAuthHeaders()
    });
    if (res.status === 401 || res.status === 403) {
      showAuthModal("Authentication required to view subscriber directory.");
      throw new Error("Authentication required (401/403)");
    }
    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      throw new Error(errData?.detail || errData?.error || "Failed to load directory");
    }
    const data = await res.json();
    renderSubscriberTable(data);
  } catch (err) {
    console.error("Failed to load subscribers directory:", err);
    const tbody = document.getElementById("directory-tbody");
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2.5rem; color: var(--rose);">⚠️ ${err.message}</td></tr>`;
  }
}

function renderSubscriberTable(data) {
  const tbody = document.getElementById("directory-tbody");
  if (!tbody) return;

  if (!data.customers || data.customers.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 2.5rem; color: var(--text-muted);">No customer accounts match your search filters.</td></tr>`;
    return;
  }

  currentDirectoryRecords = data.customers || [];

  tbody.innerHTML = data.customers.map((c, idx) => {
    const isChurn = String(c.Churn).toLowerCase() === "yes" || String(c.Churn) === "1";
    return `
      <tr>
        <td>
          <div class="cust-name-cell">
            <div class="mini-avatar" style="background: ${c.avatar_color || '#2563eb'};">${c.initials || 'C'}</div>
            <div>
              <div style="font-weight: 700; color: var(--text-main);">${c.full_name || 'Subscriber Account'}</div>
              <div style="font-size: 0.74rem; color: var(--text-faint);">${c.email || c.customerID}</div>
            </div>
          </div>
        </td>
        <td><span style="font-size: 0.78rem; padding: 2px 7px; background: var(--bg-subtle); border: 1px solid var(--border-color); border-radius: 4px;">${c.Contract}</span></td>
        <td>${c.InternetService || 'None'}</td>
        <td style="font-weight: 600;">$${parseFloat(c.MonthlyCharges || 0).toFixed(2)}</td>
        <td>
          <span style="font-size: 0.76rem; color: ${isChurn ? 'var(--rose)' : 'var(--emerald)'}; font-weight:600;">
            ${c.sentiment || (isChurn ? 'At-Risk' : 'Satisfied')}
          </span>
        </td>
        <td>
          <span style="display:inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; background: ${c.health_score > 60 ? 'var(--emerald-bg)' : 'var(--rose-bg)'}; color: ${c.health_score > 60 ? 'var(--emerald)' : 'var(--rose)'}; border: 1px solid ${c.health_score > 60 ? 'var(--emerald-border)' : 'var(--rose-border)'};">
            ${c.health_score || (isChurn ? '24' : '88')}/100
          </span>
        </td>
        <td style="font-size: 0.78rem; color: var(--text-muted);">${c.assigned_agent || 'Alex Rivera'}</td>
        <td>
          <button class="btn-outline-action btn-open-cust-account" data-index="${idx}">
            Open Account 👤
          </button>
        </td>
      </tr>
    `;
  }).join("");

  const recordsInfo = document.getElementById("directory-records-info");
  if (recordsInfo) recordsInfo.textContent = `Page ${data.page} of ${data.pages || 1} (${data.total.toLocaleString()} total accounts)`;

  const prevBtn = document.getElementById("btn-dir-prev");
  const nextBtn = document.getElementById("btn-dir-next");
  if (prevBtn) prevBtn.disabled = data.page <= 1;
  if (nextBtn) nextBtn.disabled = data.page >= data.pages;
}

function prevDirectoryPage() {
  if (crmPageNumber > 1) {
    crmPageNumber--;
    loadSubscriberDirectory();
  }
}

function nextDirectoryPage() {
  crmPageNumber++;
  loadSubscriberDirectory();
}

function openCustomerAccount(cust) {
  // Synthesize persona object
  const persona = {
    full_name: cust.full_name || "Subscriber",
    email: cust.email || `${cust.customerID}@customermail.net`,
    phone: cust.phone || "+1 (555) 438-9120",
    initials: cust.initials || "C",
    avatar_color: cust.avatar_color || "#3b82f6",
    assigned_agent: cust.assigned_agent || "Alex Rivera",
    sentiment: cust.sentiment || "Under Evaluation",
    sentiment_class: cust.sentiment?.includes("At-Risk") ? "at-risk" : "satisfied",
    last_touchpoint: cust.last_touchpoint || "3 days ago • Account Check",
    profile: cust
  };

  currentAccountData = persona;

  // Update UI Card Identity
  document.getElementById("cust-display-name").textContent = persona.full_name;
  document.getElementById("cust-display-email").textContent = persona.email;
  document.getElementById("cust-display-phone").textContent = persona.phone;
  
  const avatarBox = document.getElementById("cust-avatar-box");
  if (avatarBox) {
    avatarBox.textContent = persona.initials;
    avatarBox.style.backgroundColor = persona.avatar_color;
  }
  
  const sentimentEl = document.getElementById("cust-display-sentiment");
  if (sentimentEl) {
    sentimentEl.textContent = persona.sentiment;
    sentimentEl.className = `sentiment-badge ${persona.sentiment_class}`;
  }

  document.getElementById("cust-display-touch").textContent = persona.last_touchpoint;

  // Populate Form Fields
  setVal("input-gender", cust.gender || "Female");
  setVal("input-senior", cust.SeniorCitizen || 0);
  setVal("input-partner", cust.Partner || "No");
  setVal("input-dependents", cust.Dependents || "No");
  
  const tenure = cust.tenure || 1;
  setVal("input-tenure", tenure);
  const valTenure = document.getElementById("val-tenure");
  if (valTenure) valTenure.textContent = `${tenure} mo`;

  setVal("input-phoneservice", cust.PhoneService || "Yes");
  setVal("input-multiplelines", cust.MultipleLines || "No");
  setVal("input-internetservice", cust.InternetService || "Fiber optic");
  setVal("input-security", cust.OnlineSecurity || "No");
  setVal("input-backup", cust.OnlineBackup || "No");
  setVal("input-protection", cust.DeviceProtection || "No");
  setVal("input-techsupport", cust.TechSupport || "No");
  setVal("input-streamingtv", cust.StreamingTV || "No");
  setVal("input-streamingmovies", cust.StreamingMovies || "No");
  setVal("input-contract", cust.Contract || "Month-to-month");
  setVal("input-paperless", cust.PaperlessBilling || "Yes");
  setVal("input-payment", cust.PaymentMethod || "Electronic check");

  const monthly = parseFloat(cust.MonthlyCharges) || 50;
  setVal("input-monthly", monthly);
  const valMonthly = document.getElementById("val-monthly");
  if (valMonthly) valMonthly.textContent = `$${monthly.toFixed(2)}`;

  setVal("input-total", cust.TotalCharges || (tenure * monthly).toFixed(2));

  // Switch to Workspace Tab
  document.getElementById("nav-workspace").click();
  assessCustomerHealth();
}

// Batch CSV Processing
async function handleBatchFileSelect(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];

  const formData = new FormData();
  formData.append("file", file);

  const statusEl = document.getElementById("batch-file-status");
  if (statusEl) statusEl.textContent = `Analyzing ${file.name}...`;

  try {
    const res = await fetch(`${API_BASE}/api/batch-predict`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData
    });

    if (res.status === 401 || res.status === 403) {
      showAuthModal("Authentication required for batch analysis.");
      throw new Error("Authentication required (401/403)");
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => null);
      throw new Error(errData?.detail || errData?.error || "Batch processing failed");
    }
    const data = await res.json();

    if (statusEl) {
      statusEl.innerHTML = `✅ Successfully audited <strong>${data.total_processed}</strong> accounts: identified <strong>${data.high_risk_count}</strong> high-risk accounts (${data.churn_rate_batch}% churn propensity).`;
    }

    renderBatchTable(data.records);
  } catch (err) {
    if (statusEl) statusEl.innerHTML = `<span style="color:var(--rose);">⚠️ CSV Processing Error: ${err.message}</span>`;
  }
}

function renderBatchTable(records) {
  const tbody = document.getElementById("batch-results-tbody");
  if (!tbody || !records) return;

  tbody.innerHTML = records.map(r => `
    <tr>
      <td style="font-weight:700; color:var(--primary);">${r.customerID}</td>
      <td>${r.tenure} mo</td>
      <td>${r.Contract}</td>
      <td>$${parseFloat(r.MonthlyCharges).toFixed(2)}</td>
      <td style="font-weight: 700; color: ${r.predicted_churn === 'Yes' ? 'var(--rose)' : 'var(--emerald)'};">
        ${Math.round(r.churn_probability * 100)}%
      </td>
      <td>
        <span style="padding: 2px 8px; border-radius: 9999px; font-size: 0.74rem; font-weight: 700; background: ${r.predicted_churn === 'Yes' ? 'var(--rose-bg)' : 'var(--emerald-bg)'}; color: ${r.predicted_churn === 'Yes' ? 'var(--rose)' : 'var(--emerald)'}; border: 1px solid ${r.predicted_churn === 'Yes' ? 'var(--rose-border)' : 'var(--emerald-border)'};">
          ${r.predicted_churn === 'Yes' ? 'High Risk' : 'Low Risk'}
        </span>
      </td>
      <td style="font-size:0.8rem; color:var(--text-main);">${r.cluster_segment}</td>
    </tr>
  `).join("");
}

// Demo JSON Testing Functions
async function loadDemoJsonFile() {
  try {
    const res = await fetch(`${API_BASE}/api/demo-data`);
    if (!res.ok) throw new Error("Could not fetch demo customer JSON");
    const data = await res.json();
    openCustomerAccount(data);
  } catch (err) {
    console.error("Demo JSON load error:", err);
    alert("Could not load demo JSON: " + err.message);
  }
}

function handleSingleJsonUpload(input) {
  if (!input.files || input.files.length === 0) return;
  const file = input.files[0];
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = JSON.parse(e.target.result);
      const custData = Array.isArray(parsed) ? parsed[0] : (parsed.customer || parsed);
      openCustomerAccount(custData);
    } catch (err) {
      alert("Invalid JSON file: " + err.message);
    }
  };
  reader.readAsText(file);
}

async function loadDemoBatchJson() {
  const statusEl = document.getElementById("batch-file-status");
  if (statusEl) statusEl.textContent = "Loading demo batch JSON dataset (10 accounts)...";

  try {
    const res = await fetch(`${API_BASE}/api/demo-batch`);
    if (!res.ok) throw new Error("Could not fetch demo batch dataset");
    const batchData = await res.json();

    const blob = new Blob([JSON.stringify(batchData)], { type: "application/json" });
    const formData = new FormData();
    formData.append("file", blob, "demo_batch_customers.json");

    const predRes = await fetch(`${API_BASE}/api/batch-predict`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData
    });

    if (predRes.status === 401 || predRes.status === 403) {
      showAuthModal("Authentication required for demo batch analysis.");
      throw new Error("Authentication required (401/403)");
    }

    if (!predRes.ok) {
      const errData = await predRes.json().catch(() => null);
      throw new Error(errData?.detail || errData?.error || "Batch processing failed");
    }
    const data = await predRes.json();

    if (statusEl) {
      statusEl.innerHTML = `✅ Successfully audited <strong>${data.total_processed}</strong> accounts: identified <strong>${data.high_risk_count}</strong> high-risk accounts (${data.churn_rate_batch}% churn propensity).`;
    }

    renderBatchTable(data.records);
  } catch (err) {
    if (statusEl) statusEl.innerHTML = `<span style="color:var(--rose);">⚠️ Demo Batch Error: ${err.message}</span>`;
  }
}
