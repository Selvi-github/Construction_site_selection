import React, { useMemo, useState } from "react";
import { translations, languages } from "./i18n.js";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000";

function useTranslator(lang) {
  return useMemo(() => translations[lang] || translations.en, [lang]);
}

function App() {
  const [lang, setLang] = useState("en");
  const t = useTranslator(lang);

  const [authMode, setAuthMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState(localStorage.getItem("jwt") || "");

  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [buildingType, setBuildingType] = useState("House");
  const [floors, setFloors] = useState(2);
  const [result, setResult] = useState(null);
  const [reportEmail, setReportEmail] = useState("");
  const [status, setStatus] = useState("");

  const headers = token
    ? { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }
    : { "Content-Type": "application/json" };

  const handleAuth = async () => {
    setStatus("");
    const url = authMode === "login" ? "/api/auth/login" : "/api/auth/register";
    const res = await fetch(`${API_BASE}${url}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(`${t.error}: ${data.error || "Auth failed"}`);
      return;
    }
    if (data.access_token) {
      localStorage.setItem("jwt", data.access_token);
      setToken(data.access_token);
    }
    if (authMode === "register") {
      setStatus(t.success + ": registered");
    }
  };

  const handleAnalyze = async () => {
    setStatus("");
    if (!token) {
      setStatus("Please login first.");
      return;
    }
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        lat: Number(lat),
        lon: Number(lon),
        building_type: buildingType,
        floors: Number(floors)
      })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(`${t.error}: ${data.error || "Analyze failed"}`);
      return;
    }
    setResult(data.result);
  };

  const handleReportDownload = async () => {
    if (!result) return;
    setStatus("");
    if (!token) {
      setStatus("Please login first.");
      return;
    }
    const res = await fetch(`${API_BASE}/api/report`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        inputs: { lat: Number(lat), lon: Number(lon), building_type: buildingType, floors: Number(floors) },
        result
      })
    });
    if (!res.ok) {
      const data = await res.json();
      setStatus(`${t.error}: ${data.error || "Report failed"}`);
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const reportWindow = window.open(url, "_blank");
    const a = document.createElement("a");
    a.href = url;
    a.download = "site_feasibility_report.pdf";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
    if (!reportWindow) {
      setStatus("Report downloaded. Allow pop-ups to auto-open the report.");
      return;
    }
    setStatus("Report downloaded and opened.");
  };

  const handleReportEmail = async () => {
    if (!result) return;
    setStatus("");
    if (!token) {
      setStatus("Please login first.");
      return;
    }
    const res = await fetch(`${API_BASE}/api/report/email`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        to_email: reportEmail,
        inputs: { lat: Number(lat), lon: Number(lon), building_type: buildingType, floors: Number(floors) },
        result
      })
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus(`${t.error}: ${data.error || "Email failed"}`);
      return;
    }
    setStatus(t.success + ": emailed");
  };

  const handleLogout = () => {
    localStorage.removeItem("jwt");
    setToken("");
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Site Intelligence • India</p>
          <h1>{t.title}</h1>
          <p className="sub">Production-ready assessment with compliance gating and engineer review.</p>
        </div>
        <div className="lang">
          <label>{t.language}</label>
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            {languages.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </header>

      <section className="panel auth">
        <div className="auth-tabs">
          <button className={authMode === "login" ? "active" : ""} onClick={() => setAuthMode("login")}>{t.login}</button>
          <button className={authMode === "register" ? "active" : ""} onClick={() => setAuthMode("register")}>{t.register}</button>
        </div>
        <div className="auth-grid">
          <input placeholder={t.email} value={email} onChange={(e) => setEmail(e.target.value)} />
          <input type="password" placeholder={t.password} value={password} onChange={(e) => setPassword(e.target.value)} />
          <button onClick={handleAuth}>{authMode === "login" ? t.login : t.register}</button>
          {token && (
            <button className="ghost" onClick={handleLogout}>{t.logout}</button>
          )}
        </div>
      </section>

      <section className="panel grid">
        <div>
          <label>{t.lat}</label>
          <input value={lat} onChange={(e) => setLat(e.target.value)} />
        </div>
        <div>
          <label>{t.lon}</label>
          <input value={lon} onChange={(e) => setLon(e.target.value)} />
        </div>
        <div>
          <label>{t.buildingType}</label>
          <select value={buildingType} onChange={(e) => setBuildingType(e.target.value)}>
            <option>House</option>
            <option>Apartment</option>
            <option>School</option>
            <option>Hospital</option>
            <option>Factory</option>
            <option>Warehouse</option>
            <option>Bridge</option>
            <option>Mall</option>
          </select>
        </div>
        <div>
          <label>{t.floors}</label>
          <input type="number" value={floors} onChange={(e) => setFloors(e.target.value)} />
        </div>
        <button className="primary" onClick={handleAnalyze}>{t.analyze}</button>
      </section>

      {result && (
        <section className="panel result">
          <h2>{t.result}</h2>
          <div className="result-grid">
            <div>
              <p className="label">{t.feasibility}</p>
              <p className="value">{result.feasibility_score}%</p>
            </div>
            <div>
              <p className="label">{t.risk}</p>
              <p className="value">{result.risk_level}</p>
            </div>
            <div>
              <p className="label">{t.foundation}</p>
              <p className="value">{result.foundation}</p>
            </div>
          </div>
          <div className="actions">
            <button onClick={handleReportDownload}>{t.report}</button>
            <div className="email-box">
              <input placeholder={t.recipient} value={reportEmail} onChange={(e) => setReportEmail(e.target.value)} />
              <button onClick={handleReportEmail}>{t.emailReport}</button>
            </div>
          </div>
        </section>
      )}

      {status && <div className="status">{status}</div>}
    </div>
  );
}

export default App;
