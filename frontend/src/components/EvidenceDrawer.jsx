import React from 'react';
import { X, BookOpen, ExternalLink } from 'lucide-react';

export function EvidenceDrawer({ evidence, isOpen, onClose }) {
  if (!isOpen || !evidence) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-full sm:w-96 bg-white dark:bg-slate-900 shadow-2xl border-l border-slate-200 dark:border-slate-800 z-50 flex flex-col transform transition-transform duration-300">
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between bg-slate-50 dark:bg-slate-950">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-signal-teal" />
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
            Citation Evidence: {evidence.id}
          </h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-md text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="p-5 flex-1 overflow-y-auto space-y-4 text-xs">
        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Source Document</span>
          <p className="font-semibold text-slate-800 dark:text-slate-200 mt-0.5">{evidence.source}</p>
        </div>

        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Section</span>
          <p className="font-medium text-slate-700 dark:text-slate-300 mt-0.5">{evidence.section}</p>
        </div>

        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Relevance Score</span>
          <p className="font-mono text-emerald-600 dark:text-emerald-400 font-bold mt-0.5">
            {(evidence.score * 100).toFixed(1)}% match
          </p>
        </div>

        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Matched Query</span>
          <p className="font-mono bg-slate-100 dark:bg-slate-800 p-2 rounded text-slate-600 dark:text-slate-300 mt-0.5 text-[11px]">
            {evidence.query}
          </p>
        </div>

        <div>
          <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Excerpted Passage</span>
          <div className="mt-1 p-3 rounded-lg bg-teal-50/50 dark:bg-teal-950/20 border border-teal-200/50 dark:border-teal-900/40 text-slate-800 dark:text-slate-200 leading-relaxed font-sans">
            "{evidence.passage}"
          </div>
        </div>
      </div>
    </div>
  );
}
