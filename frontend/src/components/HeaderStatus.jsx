import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Database, Radio, Sun, Moon, ShieldAlert } from 'lucide-react';

export function HeaderStatus({ isDark, onToggleDark, reviewerName, onReviewerChange }) {
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    fetch('/api/models/status')
      .then(res => res.json())
      .then(data => setModelStatus(data))
      .catch(err => console.error("Failed to load models status:", err));
  }, []);

  return (
    <header className="bg-ink-teal text-white border-b border-ink-light px-6 py-3 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-md">
      <div className="flex items-center gap-3">
        <div className="bg-signal-teal text-white p-2 rounded-lg font-bold flex items-center justify-center">
          🩺
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight">MedAgent</h1>
            <span className="text-xs bg-signal-teal/30 text-teal-200 border border-teal-400/30 px-2 py-0.5 rounded-full font-mono uppercase tracking-wider">
              Research Prototype
            </span>
          </div>
          <p className="text-xs text-teal-200/80">Multimodal Multi-Agent Clinical Decision Support</p>
        </div>
      </div>

      {/* Model Readiness Strip */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5 bg-ink-dark/60 border border-teal-500/20 px-3 py-1.5 rounded-md">
          <Activity className="w-3.5 h-3.5 text-signal-teal" />
          <span className="text-teal-200">Imaging:</span>
          <span className={modelStatus?.imaging_model?.ready ? "text-emerald-400 font-medium" : "text-amber-400 font-medium"}>
            {modelStatus?.imaging_model?.ready ? "DenseNet121 Ready" : "Standby"}
          </span>
        </div>

        <div className="flex items-center gap-1.5 bg-ink-dark/60 border border-teal-500/20 px-3 py-1.5 rounded-md">
          <Cpu className="w-3.5 h-3.5 text-signal-teal" />
          <span className="text-teal-200">Reasoning:</span>
          <span className="text-emerald-400 font-medium uppercase font-mono">
            {modelStatus?.llm_engine?.provider || "offline rule engine"}
          </span>
        </div>

        <div className="flex items-center gap-1.5 bg-ink-dark/60 border border-teal-500/20 px-3 py-1.5 rounded-md">
          <Database className="w-3.5 h-3.5 text-signal-teal" />
          <span className="text-teal-200">RAG Store:</span>
          <span className="text-emerald-400 font-medium uppercase font-mono">
            {modelStatus?.rag_backend?.backend || "numpy"}
          </span>
        </div>
      </div>

      {/* Physician Settings & Theme */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-xs">
          <label htmlFor="physician-name" className="text-teal-200">Physician:</label>
          <input
            id="physician-name"
            type="text"
            value={reviewerName}
            onChange={(e) => onReviewerChange(e.target.value)}
            placeholder="Dr. Name"
            className="bg-ink-dark/70 border border-teal-400/40 rounded px-2.5 py-1 text-white text-xs placeholder:text-teal-300/40 focus:outline-none focus:border-signal-teal w-32"
          />
        </div>

        <button
          onClick={onToggleDark}
          title={isDark ? "Switch to light theme" : "Switch to dark theme"}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          className="p-1.5 rounded-md bg-ink-dark/60 hover:bg-ink-dark border border-teal-500/20 text-teal-200 transition-colors"
        >
          {isDark ? <Sun className="w-4 h-4 text-amber-300" /> : <Moon className="w-4 h-4 text-teal-200" />}
        </button>
      </div>
    </header>
  );
}
