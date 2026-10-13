import React, { useState } from 'react';
import { CheckCircle2, Clock, XCircle, MinusCircle, ChevronDown, ChevronRight, PlayCircle, GitFork } from 'lucide-react';

const STEP_DEFINITIONS = [
  { id: 'plan', title: 'Orchestrator Plan', desc: 'Analyzes multimodal inputs & builds execution DAG' },
  { id: 'parallel_group', title: 'Specialist Agents (Parallel)', isGroup: true },
  { id: 'rag', title: 'RAG Knowledge Retrieval', desc: 'Queries medical knowledge base for evidence' },
  { id: 'reasoning', title: 'Clinical Reasoning Agent', desc: 'Fuses findings into cited differential draft' },
  { id: 'safety', title: 'Safety & Consistency Guardrails', desc: 'Checks contraindications, dosage & urgency' },
  { id: 'physician_gate', title: 'Physician Review Gate', desc: 'Human-in-the-loop attestation and decision' },
];

export function PipelineStepper({ nodeStates }) {
  const [expandedNodes, setExpandedNodes] = useState({});

  const toggleExpand = (id) => {
    setExpandedNodes(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const renderStatusBadge = (node) => {
    const status = node?.status || 'waiting';
    switch (status) {
      case 'running':
        return (
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-signal-teal animate-ping" />
            Running
          </span>
        );
      case 'done':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
            <CheckCircle2 className="w-3 h-3 text-ok-sage" />
            Done {node.duration ? `(${node.duration.toFixed(2)}s)` : ''}
          </span>
        );
      case 'skipped':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400">
            <MinusCircle className="w-3 h-3" />
            Skipped
          </span>
        );
      case 'failed':
      case 'error':
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300">
            <XCircle className="w-3 h-3 text-urgent-coral" />
            Failed
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium text-slate-400 dark:text-slate-500">
            <Clock className="w-3 h-3" />
            Waiting
          </span>
        );
    }
  };

  const renderNodeCard = (id, title, desc) => {
    const node = nodeStates[id] || { status: 'waiting' };
    const isExpanded = expandedNodes[id];

    return (
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-3 text-xs shadow-sm transition-all">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <button
              onClick={() => toggleExpand(id)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              disabled={!node.payload}
            >
              {node.payload ? (
                isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />
              ) : (
                <div className="w-3.5 h-3.5" />
              )}
            </button>
            <div>
              <h4 className="font-semibold text-slate-800 dark:text-slate-200">{title}</h4>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{desc}</p>
            </div>
          </div>
          <div>{renderStatusBadge(node)}</div>
        </div>

        {/* One line summary preview */}
        {node.summary && (
          <div className="mt-2 text-[11px] bg-slate-50 dark:bg-slate-800/60 p-1.5 rounded text-slate-600 dark:text-slate-300 font-mono truncate">
            {node.summary}
          </div>
        )}

        {/* Expandable typed payload */}
        {isExpanded && node.payload && (
          <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-800">
            <pre className="text-[10px] bg-slate-950 text-emerald-400 p-2.5 rounded overflow-x-auto max-h-48 font-mono">
              {JSON.stringify(node.payload, null, 2)}
            </pre>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="w-full lg:w-80 flex-shrink-0 bg-paper-subtle/50 dark:bg-slate-950/40 p-4 border-r border-slate-200 dark:border-slate-800 flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-60px)]">
      <div className="flex items-center justify-between mb-1">
        <h3 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300 flex items-center gap-1.5">
          <GitFork className="w-3.5 h-3.5 text-signal-teal" />
          <span>Live Pipeline Stepper</span>
        </h3>
        <span className="text-[10px] text-slate-500 font-mono">LangGraph</span>
      </div>

      <div className="space-y-3">
        {/* Step 1: Plan */}
        {renderNodeCard('plan', 'Orchestrator Plan', 'Intake routing & DAG plan')}

        {/* Step 2: Parallel Imaging || Lab */}
        <div className="border border-teal-200/60 dark:border-teal-900/40 bg-teal-50/20 dark:bg-teal-950/20 rounded-lg p-2 space-y-2">
          <div className="text-[10px] font-bold text-teal-800 dark:text-teal-300 uppercase tracking-wider flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-signal-teal" />
            Parallel Execution Branch
          </div>
          <div className="space-y-2">
            {renderNodeCard('imaging', 'Imaging Agent (CXR)', 'DenseNet-121 + Grad-CAM')}
            {renderNodeCard('lab', 'Lab Agent', 'Deterministic parser & patterns')}
          </div>
        </div>

        {/* Step 3: RAG */}
        {renderNodeCard('rag', 'RAG Agent', 'SentenceTransformers knowledge retrieval')}

        {/* Step 4: Reasoning */}
        {renderNodeCard('reasoning', 'Clinical Reasoning', 'Multimodal fusion & differential ranking')}

        {/* Step 5: Safety */}
        {renderNodeCard('safety', 'Safety Agent', 'Grounding, contraindications & urgency')}

        {/* Step 6: Physician Gate */}
        {renderNodeCard('physician_gate', 'Physician Review Gate', 'LangGraph interrupt & attestation')}
      </div>
    </div>
  );
}
