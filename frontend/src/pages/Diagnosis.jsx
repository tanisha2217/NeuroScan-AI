import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, Activity, RefreshCw, ChevronRight, Download,
  Info, User, Microscope, Eye, Crosshair, Target, Ruler,
  AlertTriangle, CheckCircle, BarChart3
} from 'lucide-react';
import { jsPDF } from 'jspdf';

// Color map for tumor types
const TYPE_COLORS = {
  glioma_tumor:     { border: 'border-red-500',    bg: 'bg-red-500/10',    text: 'text-red-400',    hex: '#ef4444' },
  meningioma_tumor: { border: 'border-orange-500', bg: 'bg-orange-500/10', text: 'text-orange-400', hex: '#f97316' },
  pituitary_tumor:  { border: 'border-purple-500', bg: 'bg-purple-500/10', text: 'text-purple-400', hex: '#a855f7' },
  no_tumor:         { border: 'border-teal-500',   bg: 'bg-teal-500/10',   text: 'text-teal-400',   hex: '#14b8a6' },
};

const Diagnosis = () => {
  const [file, setFile]             = useState(null);
  const [preview, setPreview]       = useState(null);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [doctorId, setDoctorId]     = useState('');

  const typeStyle = result ? (TYPE_COLORS[result.tumor_type] || TYPE_COLORS.no_tumor) : null;

  const onFileChange = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setError(null);
      setShowHeatmap(false);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const fd = new FormData();
    fd.append('image', file);

    try {
      const res = await fetch('http://localhost:5000/predict', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Prediction failed');
      }
      const data = await res.json();
      // small delay for animation
      setTimeout(() => { setResult(data); setLoading(false); }, 2200);
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null); setPreview(null); setResult(null);
    setError(null); setShowHeatmap(false);
    setPatientName(''); setDoctorId('');
  };

  const handleDownloadReport = () => {
    if (!result) return;
    const doc  = new jsPDF();
    const date = new Date().toLocaleString();

    // Header
    doc.setFillColor(5, 7, 10);
    doc.rect(0, 0, 210, 40, 'F');
    doc.setTextColor(14, 165, 233);
    doc.setFontSize(22);
    doc.text("NEUROSCAN AI", 20, 25);
    doc.setFontSize(10);
    doc.setTextColor(100, 100, 100);
    doc.text("Professional AI Diagnostic Report", 20, 32);

    // Patient details
    doc.setTextColor(0, 0, 0);
    doc.setFontSize(12);
    doc.text(`Patient Name:    ${patientName || "Not Specified"}`, 20, 55);
    doc.text(`Clinician ID:    ${doctorId   || "Not Specified"}`, 20, 62);
    doc.text(`Date of Analysis: ${date}`,                          20, 69);
    doc.line(20, 75, 190, 75);

    // Findings
    doc.setFontSize(16);
    doc.text("DIAGNOSTIC FINDINGS", 20, 90);
    doc.setFontSize(12);
    doc.text(`AI Diagnosis:     ${result.diagnosis}`,          20, 105);
    doc.text(`Confidence Score: ${result.confidence}%`,        20, 112);
    doc.text(`Localization:     ${result.localization}`,       20, 119);
    if (result.bbox?.diameter) {
      doc.text(`Est. Diameter:    ${result.bbox.diameter} cm`, 20, 126);
    }
    doc.text(`Description: ${result.description}`, 20, 136, { maxWidth: 170 });

    // Scan image
    const img = (showHeatmap && result.heatmap) ? `data:image/jpeg;base64,${result.heatmap}` : preview;
    if (img) doc.addImage(img, 'JPEG', 40, 150, 130, 130);

    // Disclaimer
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text("DISCLAIMER: Educational AI tool only. Not for clinical decisions.", 20, 287);
    doc.save(`NeuroScan_${patientName || 'Report'}.pdf`);
  };

  return (
    <div className="pt-32 pb-20 px-6 max-w-6xl mx-auto min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
        className="w-full glass-panel rounded-3xl overflow-hidden flex flex-col md:flex-row min-h-[600px] border-white/5 shadow-2xl"
      >
        {/* ── Sidebar ── */}
        <div className="w-full md:w-80 bg-white/5 p-8 border-b md:border-b-0 md:border-r border-white/10 flex flex-col gap-8">
          <div className="space-y-4">
            <h1 className="text-xl font-bold tracking-tight">Diagnostic Input</h1>
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-[10px] text-gray-500 uppercase tracking-widest flex items-center gap-2">
                  <User size={12} /> Patient Name
                </label>
                <input type="text" value={patientName} onChange={e => setPatientName(e.target.value)}
                  placeholder="e.g. John Doe"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-sky-500 transition-all" />
              </div>
              <div className="space-y-2">
                <label className="text-[10px] text-gray-500 uppercase tracking-widest flex items-center gap-2">
                  <Microscope size={12} /> Clinician ID
                </label>
                <input type="text" value={doctorId} onChange={e => setDoctorId(e.target.value)}
                  placeholder="e.g. DR-AX7"
                  className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-sky-500 transition-all" />
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="p-5 bg-white/5 rounded-2xl border border-white/10 space-y-3">
            <p className="text-[10px] text-gray-500 uppercase tracking-widest font-bold">Detectable Classes</p>
            {Object.entries(TYPE_COLORS).map(([key, s]) => (
              <div key={key} className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${s.bg.replace('/10','')}`} style={{backgroundColor: s.hex}} />
                <span className={`text-xs ${s.text} capitalize`}>{key.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>

          <div className="p-4 bg-sky-500/10 rounded-2xl border border-sky-500/20">
            <div className="flex items-center gap-2 text-sky-400 mb-2">
              <Target size={16} /><span className="text-xs font-bold uppercase tracking-wider">2-Stage AI</span>
            </div>
            <p className="text-[10px] text-gray-400 leading-relaxed">
              First detects if a tumor exists, then classifies its type: Glioma, Meningioma, or Pituitary.
            </p>
          </div>
        </div>

        {/* ── Workspace ── */}
        <div className="flex-1 p-8 flex flex-col bg-black/20 overflow-hidden gap-6">
          <AnimatePresence mode="wait">
            {!preview ? (
              <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex-1 flex items-center justify-center">
                <label className="w-full max-w-sm aspect-square glass-card rounded-3xl flex flex-col items-center justify-center cursor-pointer border-dashed border-2 border-white/10 hover:border-sky-500/50 transition-all group">
                  <input type="file" className="hidden" onChange={onFileChange} accept="image/*" />
                  <div className="p-5 bg-sky-500/10 rounded-2xl mb-4 group-hover:scale-110 transition-transform">
                    <Upload className="text-sky-400 w-8 h-8" />
                  </div>
                  <span className="text-gray-300 font-bold">Select Brain Scan</span>
                  <span className="text-[10px] text-gray-500 mt-2 uppercase tracking-widest">MRI / CT Scan</span>
                </label>
              </motion.div>
            ) : (
              <motion.div key="preview" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-1 flex flex-col gap-6">

                {/* ── Image + Overlay ── */}
                <div className="flex-1 flex items-center justify-center min-h-[350px]">
                  <div className="relative inline-block glass-card p-1 rounded-2xl overflow-visible">
                    <img
                      src={(showHeatmap && result?.heatmap) ? `data:image/jpeg;base64,${result.heatmap}` : preview}
                      alt="Scan"
                      className="max-h-[420px] w-auto rounded-xl block transition-all duration-500"
                    />

                    {loading && <div className="scanner-line" />}
                  </div>
                </div>

                {/* ── Action Buttons ── */}
                <div className="flex gap-3">
                  {!result && !loading && (
                    <button onClick={handlePredict}
                      className="flex-1 bg-sky-600 hover:bg-sky-500 text-white font-bold py-4 rounded-2xl transition-all shadow-xl shadow-sky-600/20 flex items-center justify-center gap-2">
                      Analyze Scan <ChevronRight size={20} />
                    </button>
                  )}
                  {result && (
                    <>
                      <button onClick={handleReset}
                        className="flex-1 bg-white/5 hover:bg-white/10 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-2 border border-white/5">
                        <RefreshCw size={18} /> New Scan
                      </button>
                      <button onClick={handleDownloadReport}
                        className="flex-1 bg-teal-600 hover:bg-teal-500 text-white font-bold py-4 rounded-2xl flex items-center justify-center gap-2 shadow-xl shadow-teal-600/20">
                        <Download size={18} /> Download PDF
                      </button>
                    </>
                  )}
                </div>

                {/* ── Loading indicator ── */}
                {loading && (
                  <div className="flex items-center justify-center gap-3 text-sky-400 font-bold tracking-widest animate-pulse">
                    <Activity size={20} /> ANALYZING TISSUES...
                  </div>
                )}

                {/* ── Error ── */}
                {error && (
                  <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl flex items-center gap-3 text-red-400">
                    <AlertTriangle size={20} />
                    <p className="text-sm">{error}</p>
                  </div>
                )}

                {/* ── Results Card ── */}
                {result && (
                  <motion.div initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
                    className={`p-8 rounded-3xl glass-card border-l-4 ${typeStyle?.border} bg-white/[0.02] space-y-5`}>

                    {/* Row 1: Verdict + Confidence */}
                    <div className="flex justify-between items-start">
                      <div className="space-y-2">
                        <span className="text-[10px] text-gray-500 uppercase tracking-widest font-mono block">Diagnostic Verdict</span>
                        <h2 className={`text-3xl font-black ${typeStyle?.text}`}>{result.diagnosis}</h2>

                        {/* Badges */}
                        <div className="flex flex-wrap gap-2 mt-2">
                          <span className={`flex items-center gap-1.5 ${typeStyle?.bg} ${typeStyle?.text} ${typeStyle?.border} border text-xs font-bold px-3 py-1.5 rounded-lg`}>
                            <Crosshair size={14} /> {result.localization}
                          </span>
                          {result.bbox?.diameter && (
                            <span className="flex items-center gap-1.5 bg-white/5 text-gray-300 border border-white/10 text-xs font-bold px-3 py-1.5 rounded-lg">
                              <Ruler size={14} /> Est. {result.bbox.diameter} cm
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="text-right">
                        <span className="text-[10px] text-gray-500 uppercase tracking-widest font-mono block mb-1">Confidence</span>
                        <div className="text-4xl font-black text-white">{result.confidence}%</div>
                      </div>
                    </div>

                    {/* Confidence bar */}
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${result.confidence}%` }}
                        transition={{ duration: 0.8, ease: 'easeOut' }}
                        className="h-full rounded-full"
                        style={{ background: `linear-gradient(90deg, #0ea5e9, ${typeStyle?.hex})` }} />
                    </div>

                    {/* Description */}
                    {result.description && (
                      <div className="flex gap-3 p-4 bg-white/5 rounded-2xl border border-white/5">
                        <Info size={18} className="text-gray-400 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-gray-400 leading-relaxed">{result.description}</p>
                      </div>
                    )}

                    {/* Probability Breakdown */}
                    {result.breakdown && (
                      <div className="space-y-2">
                        <p className="text-[10px] text-gray-500 uppercase tracking-widest font-mono flex items-center gap-2">
                          <BarChart3 size={12} /> Class Probabilities
                        </p>
                        {Object.entries(result.breakdown)
                          .sort(([, a], [, b]) => b - a)
                          .map(([key, pct]) => {
                            const s = TYPE_COLORS[key] || TYPE_COLORS.no_tumor;
                            return (
                              <div key={key} className="flex items-center gap-3">
                                <span className="text-[11px] text-gray-400 w-36 capitalize">{key.replace(/_/g, ' ')}</span>
                                <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }} animate={{ width: `${pct}%` }}
                                    transition={{ duration: 0.6, ease: 'easeOut' }}
                                    className="h-full rounded-full"
                                    style={{ backgroundColor: s.hex }} />
                                </div>
                                <span className={`text-[11px] font-bold ${s.text} w-10 text-right`}>{pct}%</span>
                              </div>
                            );
                          })}
                      </div>
                    )}

                    {/* Grad-CAM toggle */}
                    {result.heatmap && (
                      <div className="flex items-center justify-between p-4 bg-sky-500/5 rounded-2xl border border-sky-500/10">
                        <div className="flex items-center gap-3">
                          <Eye size={18} className="text-sky-400" />
                          <span className="text-sm font-bold">Toggle Grad-CAM Heatmap</span>
                        </div>
                        <button onClick={() => setShowHeatmap(!showHeatmap)}
                          className={`w-14 h-7 rounded-full relative transition-colors ${showHeatmap ? 'bg-sky-500' : 'bg-white/10'}`}>
                          <div className={`absolute top-1 w-5 h-5 rounded-full bg-white transition-all ${showHeatmap ? 'left-8' : 'left-1'}`} />
                        </button>
                      </div>
                    )}
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

export default Diagnosis;
