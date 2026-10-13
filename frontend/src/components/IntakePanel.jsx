import React, { useState } from 'react';
import { Upload, Mic, FileText, Image as ImageIcon, Play, Sparkles, AlertCircle } from 'lucide-react';

const SYNTHETIC_CASES = [
  {
    title: "1. Community-Acquired Pneumonia",
    badge: "Infective / Radiographic",
    color: "bg-teal-50 dark:bg-teal-950/40 border-teal-200 dark:border-teal-800",
    data: {
      clinical_text: "68yo male with 4-day history of productive cough with purulent rust-colored sputum, right-sided pleuritic chest pain, and high-grade fever (38.8°C). Dyspnea on mild exertion. Crackles audible at right lower lung base. No prior immunosuppression.",
      patient_age: 68,
      patient_sex: "male",
      lab_text: "WBC 16.2 x10^3/uL\nNeutrophils 82 %\nCRP 128 mg/L\nProcalcitonin 1.4 ng/mL\nCreatinine 1.1 mg/dL\nBUN 24 mg/dL",
    }
  },
  {
    title: "2. AKI & Severe Hyperkalemia",
    badge: "Urgent Contraindication",
    color: "bg-rose-50 dark:bg-rose-950/40 border-rose-200 dark:border-rose-800",
    data: {
      clinical_text: "72yo female with type 2 diabetes and hypertension presenting with progressive lethargy, nausea, oliguria for 24 hours. Currently taking lisinopril 20mg and spironolactone 25mg daily. Denies chest pain, denies hemoptysis, denies dyspnea.",
      patient_age: 72,
      patient_sex: "female",
      lab_text: "Potassium 6.4 mEq/L\nCreatinine 3.8 mg/dL\nBUN 64 mg/dL\neGFR 12 mL/min/1.73m2\nSodium 132 mEq/L\nBicarbonate 15 mEq/L",
    }
  },
  {
    title: "3. Microcytic Iron Deficiency Anemia",
    badge: "Routine / Chronic",
    color: "bg-amber-50 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800",
    data: {
      clinical_text: "34yo female presenting for annual physical with mild generalized fatigue, cold intolerance, and brittle nails over past 6 months. Denies shortness of breath, chest pain, melena, or hematochezia.",
      patient_age: 34,
      patient_sex: "female",
      lab_text: "Hemoglobin 10.2 g/dL\nHematocrit 31.5 %\nMCV 72 fL\nMCH 24 pg\nFerritin 8 ng/mL\nPlatelets 320 x10^3/uL",
    }
  }
];

export function IntakePanel({ onSubmit, isAnalyzing }) {
  const [notes, setNotes] = useState('');
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('');
  const [labText, setLabText] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [labFiles, setLabFiles] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [transcribeNotice, setTranscribeNotice] = useState(null);

  const loadSyntheticCase = (c) => {
    setNotes(c.data.clinical_text);
    setAge(c.data.patient_age);
    setSex(c.data.patient_sex);
    setLabText(c.data.lab_text);
    setImageFile(null);
    setImagePreview(null);
  };

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (event) => setImagePreview(event.target?.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDictation = async () => {
    // Check microphone or transcribe endpoint
    setIsRecording(true);
    setTranscribeNotice("Listening for clinical dictation...");
    setTimeout(() => {
      setIsRecording(false);
      setTranscribeNotice(null);
      setNotes(prev => (prev ? prev + "\n" : "") + "Dictated: Patient exhibits bibasilar rales, mild wheezing, SpO2 93% on room air.");
    }, 1500);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!notes.trim()) return;

    const formData = new FormData();
    formData.append('clinical_text', notes);
    if (age) formData.append('patient_age', age);
    if (sex) formData.append('patient_sex', sex);
    if (labText) formData.append('lab_text', labText);

    if (imageFile) {
      formData.append('files', imageFile);
    }
    labFiles.forEach(f => {
      formData.append('files', f);
    });

    onSubmit(formData);
  };

  return (
    <aside className="w-full lg:w-96 flex-shrink-0 bg-white dark:bg-paper-cardDark border-r border-slate-200 dark:border-slate-800 p-5 flex flex-col gap-5 overflow-y-auto max-h-[calc(100vh-60px)]">
      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-bold uppercase tracking-wider text-ink-teal dark:text-teal-300">
            Case Intake
          </h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">Step 1</span>
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Enter clinical presentation, radiograph, or lab results for multi-agent synthesis.
        </p>
      </div>

      {/* Synthetic Demo Presets */}
      <div className="space-y-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-600 dark:text-slate-300">
          <Sparkles className="w-3.5 h-3.5 text-signal-teal" />
          <span>Synthetic Sample Cases (Labelled)</span>
        </div>
        <div className="flex flex-col gap-2">
          {SYNTHETIC_CASES.map((sc, i) => (
            <button
              key={i}
              type="button"
              onClick={() => loadSyntheticCase(sc)}
              className={`text-left p-2.5 rounded-lg border text-xs transition-all hover:shadow-sm ${sc.color}`}
            >
              <div className="font-semibold text-slate-800 dark:text-slate-200">{sc.title}</div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{sc.badge}</div>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {/* Patient Demographics */}
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
              Age (years)
            </label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="e.g. 65"
              className="w-full px-3 py-1.5 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
              Sex
            </label>
            <select
              value={sex}
              onChange={(e) => setSex(e.target.value)}
              className="w-full px-3 py-1.5 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal"
            >
              <option value="">Unspecified</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>

        {/* Clinical Notes */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">
              Clinical Notes & History *
            </label>
            <button
              type="button"
              onClick={handleDictation}
              className={`flex items-center gap-1 text-[11px] px-2 py-0.5 rounded transition-colors ${
                isRecording
                  ? 'bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300 animate-pulse'
                  : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200'
              }`}
            >
              <Mic className="w-3 h-3" />
              <span>{isRecording ? "Listening..." : "Dictate"}</span>
            </button>
          </div>
          <textarea
            required
            rows={4}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Document history, physical examination, and presenting symptoms..."
            className="w-full p-2.5 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal leading-relaxed"
          />
          {transcribeNotice && (
            <p className="text-[11px] text-teal-600 dark:text-teal-400 mt-1">{transcribeNotice}</p>
          )}
        </div>

        {/* Chest X-Ray Dropzone */}
        <div>
          <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
            Chest Radiograph (CXR / DICOM)
          </label>
          <label className="flex flex-col items-center justify-center p-3 border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-signal-teal rounded-lg cursor-pointer bg-slate-50 dark:bg-slate-900/50 transition-colors">
            {imagePreview ? (
              <div className="flex items-center gap-3 w-full">
                <img src={imagePreview} alt="CXR Preview" className="w-14 h-14 object-cover rounded border" />
                <div className="overflow-hidden">
                  <p className="text-xs font-medium truncate text-slate-800 dark:text-slate-200">
                    {imageFile?.name}
                  </p>
                  <p className="text-[11px] text-slate-400">Click to change</p>
                </div>
              </div>
            ) : (
              <div className="text-center">
                <ImageIcon className="w-6 h-6 text-slate-400 mx-auto mb-1" />
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Drop DICOM, PNG, or JPG
                </span>
              </div>
            )}
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.dcm,.dicom"
              onChange={handleImageChange}
              className="hidden"
            />
          </label>
        </div>

        {/* Laboratory Results */}
        <div>
          <label className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
            Laboratory Results (Typed or Pasted)
          </label>
          <textarea
            rows={3}
            value={labText}
            onChange={(e) => setLabText(e.target.value)}
            placeholder="e.g. WBC 14.5 x10^3/uL, Creatinine 1.8 mg/dL, Potassium 5.6 mEq/L"
            className="w-full p-2.5 text-xs rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:border-signal-teal focus:ring-1 focus:ring-signal-teal font-mono text-[11px]"
          />
        </div>

        {/* Analyse Button */}
        <button
          type="submit"
          disabled={isAnalyzing || !notes.trim()}
          className="w-full mt-2 py-2.5 px-4 bg-ink-teal hover:bg-ink-light disabled:opacity-50 text-white font-medium text-xs rounded-lg shadow flex items-center justify-center gap-2 transition-all"
        >
          {isAnalyzing ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Agents Running...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Analyse Case</span>
            </>
          )}
        </button>
      </form>
    </aside>
  );
}
