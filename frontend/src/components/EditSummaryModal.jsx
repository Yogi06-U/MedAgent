import React, { useState } from 'react';
import { X, Check } from 'lucide-react';

export function EditSummaryModal({ isOpen, initialSummary, onSave, onClose }) {
  const [editedText, setEditedText] = useState(initialSummary || '');
  const [physicianNotes, setPhysicianNotes] = useState('');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-2xl max-w-2xl w-full p-6 text-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
            Edit Clinical Summary & Impression
          </h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200">
            <X className="w-4 h-4" />
          </button>
        </div>

        <p className="text-slate-500 dark:text-slate-400">
          As the attending physician, you have full authority to rewrite or adjust the draft synthesis. The finalized report and PDF will reflect your edits.
        </p>

        <div>
          <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
            Clinical Summary (Edited)
          </label>
          <textarea
            rows={6}
            value={editedText}
            onChange={(e) => setEditedText(e.target.value)}
            className="w-full p-3 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal leading-relaxed font-sans"
          />
        </div>

        <div>
          <label className="block font-semibold text-slate-700 dark:text-slate-300 mb-1">
            Physician Justification / Rationale Notes
          </label>
          <input
            type="text"
            value={physicianNotes}
            onChange={(e) => setPhysicianNotes(e.target.value)}
            placeholder="e.g. Adjusted differential based on prior echocardiogram and lack of purulence"
            className="w-full p-2.5 text-xs rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal"
          />
        </div>

        <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100 dark:border-slate-800">
          <button
            type="button"
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg border border-slate-300 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onSave(editedText, physicianNotes)}
            className="px-4 py-1.5 rounded-lg bg-signal-teal hover:bg-signal-hover text-white font-medium flex items-center gap-1.5 shadow-sm"
          >
            <Check className="w-3.5 h-3.5" />
            <span>Save & Release with Edits</span>
          </button>
        </div>
      </div>
    </div>
  );
}
