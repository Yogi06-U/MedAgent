import React, { useState } from 'react';
import {
  FileText,
  Activity,
  FlaskConical,
  BookOpen,
  ShieldAlert,
  History,
  AlertTriangle,
  CheckCircle2,
  Info,
  Sliders,
  Eye,
  EyeOff,
  ChevronRight,
  ExternalLink,
} from 'lucide-react';

export function WorkspaceTabs({
  caseId,
  caseInput,
  draft,
  imagingReport,
  labReport,
  ragReport,
  safetyReport,
  auditEvents,
  onSelectCitation,
}) {
  const [activeTab, setActiveTab] = useState('summary');
  const [selectedFinding, setSelectedFinding] = useState(null);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.65);
  const [showOverlay, setShowOverlay] = useState(true);

  // Set default selected finding when imaging report arrives
  React.useEffect(() => {
    if (imagingReport?.findings?.length && !selectedFinding) {
      setSelectedFinding(imagingReport.findings[0].finding);
    }
  }, [imagingReport]);

  const tabs = [
    { id: 'summary', label: 'Summary & Differential', icon: FileText },
    { id: 'imaging', label: 'Imaging (CXR)', icon: Activity, badge: imagingReport?.findings?.length },
    { id: 'labs', label: 'Laboratory Results', icon: FlaskConical, badge: labReport?.results?.length },
    { id: 'evidence', label: 'Evidence & RAG', icon: BookOpen, badge: ragReport?.evidence?.length },
    { id: 'safety', label: 'Safety & Guardrails', icon: ShieldAlert, alert: safetyReport?.verdict === 'blocked' },
    { id: 'audit', label: 'Audit Trail', icon: History, badge: auditEvents?.length },
  ];

  const getLikelihoodBadge = (likelihood) => {
    switch (likelihood) {
      case 'high':
        return 'bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300 border-teal-300';
      case 'moderate':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-300';
      case 'low':
      default:
        return 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-300';
    }
  };

  const getLabFlagBadge = (flag) => {
    switch (flag) {
      case 'critical high':
      case 'critical low':
        return 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 font-bold';
      case 'high':
      case 'low':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 font-semibold';
      case 'normal':
        return 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300';
      default:
        return 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400';
    }
  };

  return (
    <main className="flex-1 flex flex-col bg-paper dark:bg-slate-900 overflow-hidden">
      {/* Tab Navigation */}
      <div className="flex items-center gap-1 border-b border-slate-200 dark:border-slate-800 px-6 pt-3 bg-white dark:bg-slate-950 overflow-x-auto">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-2 px-3.5 py-2.5 text-xs font-semibold rounded-t-lg transition-colors border-b-2 ${
                isActive
                  ? 'border-signal-teal text-ink-teal dark:text-teal-300 bg-paper dark:bg-slate-900'
                  : 'border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{t.label}</span>
              {t.badge !== undefined && (
                <span className="text-[10px] bg-slate-200 dark:bg-slate-800 px-1.5 py-0.2 rounded-full font-mono">
                  {t.badge}
                </span>
              )}
              {t.alert && (
                <span className="w-2 h-2 rounded-full bg-urgent-coral animate-ping" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Content Container */}
      <div className="flex-1 p-6 overflow-y-auto space-y-5 text-xs">
        {/* SUMMARY TAB */}
        {activeTab === 'summary' && (
          <div className="space-y-5">
            {/* Urgent Red Flag Banner */}
            {safetyReport?.urgent && (
              <div className="bg-rose-50 dark:bg-rose-950/40 border-l-4 border-urgent-coral p-4 rounded-r-lg">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-urgent-coral flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-bold text-urgent-coral">
                      Urgent Clinical Consideration Flagged
                    </h4>
                    <ul className="mt-1 list-disc list-inside text-rose-800 dark:text-rose-200 space-y-0.5">
                      {safetyReport.urgent_reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Clinical Summary */}
            <div className="bg-white dark:bg-slate-950 p-5 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                  Multimodal Clinical Impression
                </h3>
                <span className="text-[10px] text-slate-400 font-mono">
                  Mode: {draft?.mode || 'rule_engine'}
                </span>
              </div>
              <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-sans">
                {draft?.summary || "Awaiting multi-agent synthesis..."}
              </p>
            </div>

            {/* Ranked Differentials */}
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                Ranked Differential Diagnoses
              </h3>

              {draft?.differentials?.length ? (
                draft.differentials.map((diff, idx) => (
                  <div
                    key={idx}
                    className="bg-white dark:bg-slate-950 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm space-y-2.5"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-bold text-slate-400">#{idx + 1}</span>
                        <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100">
                          {diff.condition}
                        </h4>
                      </div>
                      <span className={`px-2.5 py-0.5 text-[11px] font-bold uppercase rounded-full border ${getLikelihoodBadge(diff.likelihood)}`}>
                        {diff.likelihood} Likelihood
                      </span>
                    </div>

                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                      <strong className="text-slate-900 dark:text-slate-100">Rationale: </strong>
                      {diff.rationale}
                    </p>

                    {diff.supporting_findings?.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 pt-1">
                        <span className="text-slate-500 font-medium text-[11px]">Supporting:</span>
                        {diff.supporting_findings.map((sf, sidx) => (
                          <span
                            key={sidx}
                            className="bg-teal-50 dark:bg-teal-950/60 text-teal-800 dark:text-teal-300 px-2 py-0.5 rounded border border-teal-200 dark:border-teal-800 text-[11px]"
                          >
                            {sf}
                          </span>
                        ))}
                      </div>
                    )}

                    {diff.against && (
                      <p className="text-[11px] text-slate-500 dark:text-slate-400 italic">
                        <strong className="not-italic text-slate-600 dark:text-slate-300">Discordant / Against: </strong>
                        {diff.against}
                      </p>
                    )}

                    {diff.evidence_ids?.length > 0 && (
                      <div className="flex items-center gap-2 pt-1 border-t border-slate-100 dark:border-slate-800">
                        <span className="text-[11px] text-slate-400">Citations:</span>
                        <div className="flex items-center gap-1.5">
                          {diff.evidence_ids.map((eid) => (
                            <button
                              key={eid}
                              onClick={() => {
                                const ev = ragReport?.evidence?.find((e) => e.id === eid);
                                if (ev) onSelectCitation(ev);
                              }}
                              className="px-2 py-0.5 rounded bg-signal-teal/10 hover:bg-signal-teal/20 text-signal-teal font-mono text-[11px] font-semibold flex items-center gap-1"
                            >
                              <span>{eid}</span>
                              <ExternalLink className="w-2.5 h-2.5" />
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <p className="text-slate-400 italic">No differential diagnoses available.</p>
              )}
            </div>

            {/* Suggested Workup & Red Flags */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {draft?.suggested_workup?.length > 0 && (
                <div className="bg-white dark:bg-slate-950 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300 mb-2">
                    Suggested Diagnostic Workup
                  </h4>
                  <ul className="list-disc list-inside space-y-1 text-slate-700 dark:text-slate-300">
                    {draft.suggested_workup.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

              {draft?.uncertainties?.length > 0 && (
                <div className="bg-white dark:bg-slate-950 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300 mb-2">
                    Clinical Uncertainties & Gaps
                  </h4>
                  <ul className="list-disc list-inside space-y-1 text-slate-700 dark:text-slate-300">
                    {draft.uncertainties.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* IMAGING TAB */}
        {activeTab === 'imaging' && (
          <div className="space-y-4">
            <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 p-3 rounded-lg text-amber-800 dark:text-amber-200 text-xs">
              <strong>Mandatory Notice:</strong> AI model read, not a radiologist read. Indicative severity is a heuristic based on probability and Grad-CAM spatial extent.
            </div>

            {imagingReport?.status === 'ok' ? (
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Radiograph & Overlay Viewer */}
                <div className="lg:col-span-7 bg-black rounded-xl p-4 flex flex-col items-center justify-center border border-slate-800">
                  <div className="relative w-full max-w-md aspect-square bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center">
                    {/* Base image or placeholder */}
                    {caseInput?.image_paths?.[0] ? (
                      <img
                        src={`/api/cases/${caseId}/files/${caseInput.image_paths[0].split('/').pop()}`}
                        alt="Radiograph"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="text-slate-500 font-mono text-center p-4">
                        [Chest Radiograph Base]
                      </div>
                    )}

                    {/* Grad-CAM Overlay */}
                    {showOverlay && selectedFinding && imagingReport.heatmap_paths[selectedFinding] && (
                      <img
                        src={`/api/cases/${caseId}/files/${imagingReport.heatmap_paths[selectedFinding].split('/').pop()}`}
                        alt={`Grad-CAM overlay for ${selectedFinding}`}
                        style={{ opacity: heatmapOpacity }}
                        className="absolute inset-0 w-full h-full object-contain mix-blend-screen pointer-events-none transition-opacity"
                      />
                    )}
                  </div>

                  {/* Viewer Controls */}
                  <div className="w-full max-w-md mt-4 flex items-center justify-between gap-4 text-xs text-slate-300">
                    <button
                      onClick={() => setShowOverlay(!showOverlay)}
                      className="flex items-center gap-1.5 px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded text-slate-200"
                    >
                      {showOverlay ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                      <span>{showOverlay ? "Hide Heatmap" : "Show Heatmap"}</span>
                    </button>

                    <div className="flex items-center gap-2 flex-1 max-w-[200px]">
                      <Sliders className="w-3.5 h-3.5 text-slate-400" />
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        value={heatmapOpacity}
                        onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                        className="w-full accent-signal-teal"
                      />
                      <span className="font-mono text-[10px] w-8">
                        {Math.round(heatmapOpacity * 100)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Findings List */}
                <div className="lg:col-span-5 space-y-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                    Detected Findings {imagingReport.findings?.length > 0 ? `(Top-${imagingReport.findings.length}, p ≥ 50%)` : ''}
                  </h4>

                  {(!imagingReport.findings || imagingReport.findings.length === 0) ? (
                    <div className="p-4 rounded-lg border border-teal-200 dark:border-teal-900/50 bg-teal-50/60 dark:bg-teal-950/20 text-center space-y-2">
                      <CheckCircle2 className="w-8 h-8 text-teal-600 dark:text-teal-400 mx-auto" />
                      <div className="font-bold text-sm text-teal-900 dark:text-teal-200">
                        Unremarkable Radiograph
                      </div>
                      <p className="text-xs text-teal-700 dark:text-teal-300">
                        {imagingReport.message || "No acute cardiopulmonary abnormalities detected above operating threshold."}
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {imagingReport.findings.map((f, i) => (
                        <div
                          key={i}
                          onClick={() => setSelectedFinding(f.finding)}
                          className={`p-3 rounded-lg border cursor-pointer transition-all ${
                            selectedFinding === f.finding
                              ? 'border-signal-teal bg-teal-50 dark:bg-teal-950/40 shadow-sm'
                              : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 hover:border-slate-300'
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-bold text-slate-800 dark:text-slate-100">
                              {f.finding}
                            </span>
                            <span className="font-mono font-bold text-signal-teal">
                              {(f.probability * 100).toFixed(1)}%
                            </span>
                          </div>

                          {/* Probability Progress Bar */}
                          <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden mb-2">
                            <div
                              className="bg-signal-teal h-full rounded-full"
                              style={{ width: `${f.probability * 100}%` }}
                            />
                          </div>

                          <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                            <span>Location: <strong>{f.location}</strong></span>
                            <span>Indicative Severity: <strong className="capitalize">{f.severity}</strong></span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {imagingReport.image_quality_notes?.length > 0 && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg">
                      <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Quality Notes</span>
                      <ul className="list-disc list-inside mt-1 text-slate-600 dark:text-slate-300 text-[11px]">
                        {imagingReport.image_quality_notes.map((qn, i) => (
                          <li key={i}>{qn}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center p-8 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-400">
                {imagingReport?.message || "No radiograph uploaded or analyzed for this case."}
              </div>
            )}
          </div>
        )}

        {/* LABS TAB */}
        {activeTab === 'labs' && (
          <div className="space-y-4">
            {labReport?.status === 'ok' ? (
              <>
                {/* Identified Clinical Patterns */}
                {labReport.patterns?.length > 0 && (
                  <div className="bg-white dark:bg-slate-950 p-4 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm space-y-2">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                      Synthesized Laboratory Patterns
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {labReport.patterns.map((pat, i) => (
                        <div
                          key={i}
                          className="p-3 rounded-lg border border-teal-200/60 dark:border-teal-900/40 bg-teal-50/30 dark:bg-teal-950/20"
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-bold text-slate-800 dark:text-slate-100">{pat.name}</span>
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              pat.urgency === 'urgent'
                                ? 'bg-rose-100 text-rose-800'
                                : pat.urgency === 'soon'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}>
                              {pat.urgency}
                            </span>
                          </div>
                          <p className="text-slate-600 dark:text-slate-300 text-[11px] leading-relaxed">
                            {pat.description}
                          </p>
                          {pat.supporting_tests?.length > 0 && (
                            <p className="text-[10px] text-slate-400 mt-1">
                              Supporting: {pat.supporting_tests.join(', ')}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Results Table */}
                <div className="bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
                  <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                      Standardized Results ({labReport.results.length} tests)
                    </h4>
                    <span className="text-[11px] text-slate-400">Abnormal flagged first</span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-[11px] uppercase tracking-wider text-slate-500">
                          <th className="py-2.5 px-4 font-semibold">Test Name</th>
                          <th className="py-2.5 px-4 font-semibold">Value & Unit</th>
                          <th className="py-2.5 px-4 font-semibold">Reference Interval</th>
                          <th className="py-2.5 px-4 font-semibold">Range Position</th>
                          <th className="py-2.5 px-4 font-semibold">Flag</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800 tabular-nums">
                        {labReport.results.map((res, i) => {
                          const hasRange = res.ref_low !== null && res.ref_high !== null;
                          let rangePercent = 50;
                          if (hasRange && res.ref_high > res.ref_low) {
                            rangePercent = Math.min(
                              100,
                              Math.max(
                                0,
                                ((res.value - res.ref_low) / (res.ref_high - res.ref_low)) * 100
                              )
                            );
                          }

                          return (
                            <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-850/50 transition-colors">
                              <td className="py-2.5 px-4 font-medium text-slate-800 dark:text-slate-200">
                                {res.test}
                                {res.raw_name !== res.test && (
                                  <span className="text-[10px] text-slate-400 block font-normal">
                                    raw: {res.raw_name}
                                  </span>
                                )}
                              </td>
                              <td className="py-2.5 px-4 font-mono font-bold text-slate-900 dark:text-slate-100">
                                {res.value} {res.unit}
                              </td>
                              <td className="py-2.5 px-4 text-slate-500 font-mono text-[11px]">
                                {hasRange ? `${res.ref_low} – ${res.ref_high}` : 'Not established'}
                              </td>
                              <td className="py-2.5 px-4 w-32">
                                {hasRange && (
                                  <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full relative">
                                    <div
                                      className={`h-2 rounded-full ${
                                        res.flag.includes('critical')
                                          ? 'bg-rose-500'
                                          : res.flag !== 'normal'
                                          ? 'bg-amber-500'
                                          : 'bg-emerald-500'
                                      }`}
                                      style={{ width: `${rangePercent}%` }}
                                    />
                                  </div>
                                )}
                              </td>
                              <td className="py-2.5 px-4">
                                <span className={`px-2 py-0.5 rounded text-[10px] uppercase ${getLabFlagBadge(res.flag)}`}>
                                  {res.flag}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {labReport.unparsed_lines?.length > 0 && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 text-[11px]">
                      <span className="font-semibold text-slate-500">Unparsed Lines: </span>
                      <span className="text-slate-400 font-mono">{labReport.unparsed_lines.join(', ')}</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-center p-8 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-400">
                {labReport?.message || "No laboratory data processed for this case."}
              </div>
            )}
          </div>
        )}

        {/* EVIDENCE TAB */}
        {activeTab === 'evidence' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                Retrieved Knowledge Base Passages
              </h4>
              <span className="text-slate-400 font-mono text-[11px]">
                Backend: {ragReport?.backend || 'numpy'}
              </span>
            </div>

            {ragReport?.evidence?.length ? (
              <div className="space-y-3">
                {ragReport.evidence.map((ev, i) => (
                  <div
                    key={i}
                    onClick={() => onSelectCitation(ev)}
                    className="p-4 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-signal-teal shadow-sm cursor-pointer transition-all space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded bg-teal-100 dark:bg-teal-950 text-signal-teal font-mono font-bold text-xs">
                          {ev.id}
                        </span>
                        <span className="font-bold text-slate-800 dark:text-slate-200">{ev.source}</span>
                      </div>
                      <span className="text-[11px] font-mono text-emerald-600 font-bold">
                        {(ev.score * 100).toFixed(1)}% match
                      </span>
                    </div>

                    <p className="text-slate-600 dark:text-slate-300 text-xs italic font-serif leading-relaxed">
                      "{ev.passage}"
                    </p>

                    <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between pt-1 border-t border-slate-100 dark:border-slate-850">
                      <span>Section: {ev.section}</span>
                      <span>Query: {ev.query}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center p-8 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-400">
                {ragReport?.message || "No evidence retrieved."}
              </div>
            )}
          </div>
        )}

        {/* SAFETY TAB */}
        {activeTab === 'safety' && (
          <div className="space-y-4">
            {safetyReport ? (
              <>
                <div className={`p-4 rounded-xl border flex items-center justify-between ${
                  safetyReport.verdict === 'blocked'
                    ? 'bg-rose-50 dark:bg-rose-950/40 border-rose-300 text-rose-900 dark:text-rose-200'
                    : safetyReport.verdict === 'pass_with_warnings'
                    ? 'bg-amber-50 dark:bg-amber-950/40 border-amber-300 text-amber-900 dark:text-amber-200'
                    : 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 text-emerald-900 dark:text-emerald-200'
                }`}>
                  <div className="flex items-center gap-3">
                    <ShieldAlert className="w-6 h-6" />
                    <div>
                      <h4 className="text-sm font-bold uppercase tracking-wider">
                        Safety Verdict: {safetyReport.verdict.replace('_', ' ')}
                      </h4>
                      <p className="text-xs opacity-90 mt-0.5">
                        {safetyReport.verdict === 'blocked'
                          ? 'Blocker issues detected. Automatic physician direct approval is disabled.'
                          : 'Clinical checks evaluated against contraindications and dosage guidelines.'}
                      </p>
                    </div>
                  </div>
                  <span className="text-xs font-mono font-bold uppercase px-3 py-1 rounded bg-white/60 dark:bg-black/30">
                    {safetyReport.issues?.length || 0} issues
                  </span>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
                    Audit of Checked Guardrails
                  </h4>

                  {safetyReport.issues?.length ? (
                    safetyReport.issues.map((iss, i) => (
                      <div
                        key={i}
                        className="p-3 bg-white dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 flex items-start gap-3"
                      >
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase mt-0.5 ${
                          iss.severity === 'blocker'
                            ? 'bg-rose-100 text-rose-800'
                            : iss.severity === 'warning'
                            ? 'bg-amber-100 text-amber-800'
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {iss.severity}
                        </span>
                        <div className="flex-1">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-300">
                              {iss.code}
                            </span>
                            <span className="text-[10px] text-slate-400">{iss.where}</span>
                          </div>
                          <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">
                            {iss.message}
                          </p>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="p-4 bg-white dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-500 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                      <span>All safety checks passed with zero issues or contraindications detected.</span>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-center p-8 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-400">
                Safety Agent evaluation pending...
              </div>
            )}
          </div>
        )}

        {/* AUDIT TAB */}
        {activeTab === 'audit' && (
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
              Case Event Stream & Lifecycle
            </h4>

            {auditEvents?.length ? (
              <div className="space-y-2">
                {auditEvents.map((evt, i) => (
                  <div
                    key={i}
                    className="p-3 bg-white dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-signal-teal" />
                      <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">
                        {evt.event}
                      </span>
                      {evt.data?.node && (
                        <span className="text-slate-400 font-mono text-[11px]">
                          [{evt.data.node}]
                        </span>
                      )}
                    </div>
                    <span className="text-slate-400 font-mono text-[11px]">
                      {new Date(evt.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center p-8 bg-white dark:bg-slate-950 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-400">
                No events recorded for this session yet.
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
