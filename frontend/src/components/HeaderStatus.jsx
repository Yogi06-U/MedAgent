import React from 'react';
import { Sun, Moon } from 'lucide-react';

export function HeaderStatus({ isDark, onToggleDark, reviewerName, onReviewerChange }) {
  return (
    <header className="bg-ink-teal text-white border-b border-ink-light px-6 py-3 flex flex-wrap items-center justify-between gap-4 sticky top-0 z-30 shadow-md">
      <div className="flex items-center gap-3">
        <div className="bg-signal-teal text-white p-2 rounded-lg font-bold flex items-center justify-center">
          🩺
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight">MedAgent</h1>
          </div>
          <p className="text-xs text-teal-200/80">Clinical Decision Support System</p>
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
