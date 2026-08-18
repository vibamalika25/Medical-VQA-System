import { useState, useRef, useEffect } from 'react';
import { Upload, ImagePlus, Send, Sparkles, ChevronDown, AlertCircle, CheckCircle2, Info, Flame, Activity, Stethoscope } from 'lucide-react';
import { analyzeImage, checkHealth } from '../utils/mockApi';

const exampleQuestions = [
  'What abnormality is present in the lungs?',
  'Is there any cardiomegaly visible?',
  'What type of imaging modality is this?',
  'Are there signs of pneumothorax?',
  'Is the trachea midline?',
];

export default function AnalysisSection() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const fileRef = useRef(null);

  useEffect(() => {
    checkHealth().then((h) =>
      setBackendStatus(h.status === 'ok' ? 'online' : 'offline')
    );
  }, []);

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async () => {
    if (!image || !question.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await analyzeImage(image, question);
      setResult(res);
    } catch (err) {
      setResult({
        answer: 'Analysis failed',
        confidence: 0,
        organ: 'unknown',
        diagnosis: 'unknown',
        explanation: err.message || 'An error occurred while analysing the image.',
      });
    }
    setLoading(false);
  };

  const clearAll = () => {
    setImage(null);
    setPreview(null);
    setQuestion('');
    setResult(null);
  };

  return (
    <section id="analysis" className="py-24 bg-surface relative">
      {/* Background decoration */}
      <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-white to-transparent" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-14">
          <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
            <span className={`w-2 h-2 rounded-full ${
              backendStatus === 'online' ? 'bg-green-500 animate-pulse' :
              backendStatus === 'offline' ? 'bg-amber-500' : 'bg-gray-400 animate-pulse'
            }`} />
            {backendStatus === 'online' ? 'AI Engine Online' : backendStatus === 'offline' ? 'Using Demo Mode' : 'Connecting...'}
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
            Image Analysis Interface
          </h2>
          <p className="text-text-secondary text-lg">
            Upload a medical image, ask a question, and let our AI provide diagnostic insights.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left column — Upload + Question */}
          <div className="space-y-6">
            {/* Upload card */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-50 flex items-center justify-between">
                <h3 className="font-semibold text-text-primary flex items-center gap-2">
                  <ImagePlus className="w-5 h-5 text-primary" />
                  Medical Image
                </h3>
                {image && (
                  <button onClick={clearAll} className="text-xs text-red-500 hover:text-red-700 font-medium">
                    Clear
                  </button>
                )}
              </div>
              <div className="p-6">
                {!preview ? (
                  <div
                    onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={handleDrop}
                    onClick={() => fileRef.current?.click()}
                    className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all
                      ${dragActive
                        ? 'border-primary bg-primary/5 scale-[1.02]'
                        : 'border-gray-200 hover:border-primary/40 hover:bg-primary/[0.02]'
                      }`}
                  >
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={(e) => handleFile(e.target.files[0])}
                    />
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                      <Upload className="w-8 h-8 text-primary" />
                    </div>
                    <p className="text-text-primary font-semibold mb-1">Drag & drop or click to upload</p>
                    <p className="text-text-secondary text-sm">Supports X-ray, MRI, CT scan images (PNG, JPG)</p>
                  </div>
                ) : (
                  <div className="relative group">
                    <img
                      src={preview}
                      alt="Uploaded medical scan"
                      className="w-full h-64 object-contain rounded-xl bg-gray-900"
                    />
                    {/* Heatmap overlay placeholder */}
                    {result && (
                      <div className="absolute inset-0 rounded-xl bg-gradient-to-t from-red-500/20 via-yellow-500/10 to-transparent pointer-events-none" />
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Question input card */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-50">
                <h3 className="font-semibold text-text-primary flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-accent" />
                  Ask a Question
                </h3>
              </div>
              <div className="p-6 space-y-4">
                <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What abnormality is present in the lungs?"
                  rows={3}
                  className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none resize-none text-sm transition-all"
                />

                {/* Example chips */}
                <div>
                  <p className="text-xs text-text-secondary mb-2 flex items-center gap-1">
                    <ChevronDown className="w-3 h-3" /> Example questions
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {exampleQuestions.map((q) => (
                      <button
                        key={q}
                        onClick={() => setQuestion(q)}
                        className="px-3 py-1.5 rounded-lg bg-surface text-text-secondary text-xs hover:bg-primary/10 hover:text-primary transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!image || !question.trim() || loading}
                  className="w-full py-3.5 rounded-xl bg-gradient-to-r from-primary to-accent text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-primary/20 hover:shadow-primary/30 hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
                >
                  {loading ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5" />
                      Analyze Image
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Right column — Results */}
          <div>
            {!result && !loading ? (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 h-full flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="w-20 h-20 mx-auto rounded-full bg-surface flex items-center justify-center mb-5">
                    <Info className="w-10 h-10 text-text-secondary/30" />
                  </div>
                  <h3 className="text-lg font-semibold text-text-primary mb-2">No Results Yet</h3>
                  <p className="text-text-secondary text-sm max-w-xs mx-auto">
                    Upload a medical image and ask a question to receive AI-powered diagnostic predictions.
                  </p>
                </div>
              </div>
            ) : loading ? (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 h-full flex items-center justify-center p-12">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto border-4 border-primary/20 border-t-primary rounded-full animate-spin mb-6" />
                  <h3 className="text-lg font-semibold text-text-primary mb-2">Analyzing Image...</h3>
                  <p className="text-text-secondary text-sm">Running deep learning models on your image</p>
                </div>
              </div>
            ) : (
              <div className="space-y-5 animate-fade-in-up">
                {/* Answer card */}
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-50 flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-success" />
                    <h3 className="font-semibold text-text-primary">Predicted Answer</h3>
                  </div>
                    <div className="p-6">
                    <p className="text-xl font-bold text-primary mb-3">{result.answer}</p>

                    {/* Organ & Diagnosis badges */}
                    {(result.organ || result.diagnosis) && (
                      <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-gray-100">
                        {result.organ && result.organ !== 'unknown' && (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 text-blue-700 text-xs font-semibold">
                            <Stethoscope className="w-3.5 h-3.5" />
                            Organ: {result.organ.charAt(0).toUpperCase() + result.organ.slice(1)}
                          </span>
                        )}
                        {result.diagnosis && result.diagnosis !== 'unknown' && (
                          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold ${
                            result.diagnosis === 'normal' ? 'bg-green-50 text-green-700' :
                            result.diagnosis === 'malignant' ? 'bg-red-50 text-red-700' :
                            'bg-amber-50 text-amber-700'
                          }`}>
                            <Activity className="w-3.5 h-3.5" />
                            Diagnosis: {result.diagnosis.charAt(0).toUpperCase() + result.diagnosis.slice(1)}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Explanation card */}
                <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-50 flex items-center gap-2">
                    <Flame className="w-5 h-5 text-orange-500" />
                    <h3 className="font-semibold text-text-primary">AI Explanation</h3>
                  </div>
                  <div className="p-6">
                    <p className="text-text-secondary text-sm leading-relaxed">{result.explanation}</p>
                  </div>
                </div>

                {/* Warning card */}
                <div className="rounded-2xl bg-amber-50 border border-amber-200 p-5 flex gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-amber-800">Disclaimer</p>
                    <p className="text-xs text-amber-700 mt-1">
                      This AI tool is for research and educational purposes only. Always consult a qualified
                      medical professional for clinical decisions.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
