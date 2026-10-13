import React, { useState } from 'react';
import { Check, Edit3, X, Download, ShieldAlert, FileCheck } from 'lucide-react';

export function PhysicianBar({
  caseId,
  isAwaitingPhysician,
  safetyReport,
  finalReport,
  reviewerName,
  onApprove,
  onEditClick,
  onReject,
  onDownloadPdf,
}) {
  const [rejectReason, setRejectReason] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);

  const isBlocked = safetyReport?.verdict === 'blocked';

  return (
    <div className="sticky bottom-0 bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 px-6 py-3 shadow-lg z-20 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
            Attending Physician:
          </span>
          <span className="text-xs font-mono font-bold text-ink-teal dark:text-teal-300 bg-teal-50 dark:bg-teal-950/60 px-2 py-0.5 rounded border border-teal-200 dark:border-teal-800">
            {reviewerName || "Unspecified"}
          </span>
        </div>

        {isBlocked && isAwaitingPhysician && (
          <div className="flex items-center gap-1.5 text-xs text-urgent-coral font-medium bg-rose-50 dark:bg-rose-950/50 px-3 py-1 rounded-md border border-rose-200 dark:border-rose-900">
            <ShieldAlert className="w-3.5 h-3.5" />
            <span>Draft BLOCKED by Safety Agent: Direct Approval Disabled.</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        {isAwaitingPhysician && (
          <>
            {showRejectInput ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  placeholder="Reason for rejection..."
                  value={rejectReason}
                  onChange={(e) => setRejectReason(e.target.value)}
                  className="px-2.5 py-1 text-xs border border-rose-300 dark:border-rose-800 rounded bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-200"
                />
                <button
                  onClick={() => onReject(rejectReason)}
                  className="px-3 py-1.5 bg-urgent-coral hover:bg-rose-700 text-white rounded text-xs font-medium"
                >
                  Confirm Reject
                </button>
                <button
                  onClick={() => setShowRejectInput(false)}
                  className="px-2 py-1 text-xs text-slate-500 hover:text-slate-700"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowRejectInput(true)}
                className="px-3 py-1.5 border border-rose-300 dark:border-rose-900 text-urgent-coral hover:bg-rose-50 dark:hover:bg-rose-950/40 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                <span>Reject</span>
              </button>
            )}

            <button
              onClick={onEditClick}
              className="px-3 py-1.5 border border-signal-teal text-signal-teal hover:bg-teal-50 dark:hover:bg-teal-950/40 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors"
            >
              <Edit3 className="w-3.5 h-3.5" />
              <span>Edit Summary</span>
            </button>

            {/* Direct Approve only rendered if NOT blocked */}
            {!isBlocked && (
              <button
                onClick={onApprove}
                className="px-4 py-1.5 bg-ok-sage hover:bg-emerald-700 text-white rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-sm transition-colors"
              >
                <Check className="w-3.5 h-3.5" />
                <span>Approve & Release</span>
              </button>
            )}
          </>
        )}

        {finalReport && (
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold flex items-center gap-1 text-slate-700 dark:text-slate-300">
              <FileCheck className="w-3.5 h-3.5 text-ok-sage" />
              Decision: <span className="uppercase font-mono text-signal-teal">{finalReport.decision}</span>
            </span>

            {finalReport.pdf_path && (
              <button
                onClick={onDownloadPdf}
                className="px-3.5 py-1.5 bg-ink-teal hover:bg-ink-light text-white rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-sm transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Report PDF</span>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
