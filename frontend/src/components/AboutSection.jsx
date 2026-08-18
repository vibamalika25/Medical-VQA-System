import { GraduationCap, Target, Users, FlaskConical } from 'lucide-react';

const highlights = [
  { icon: GraduationCap, label: 'Deep Learning Research', color: 'text-blue-600', bg: 'bg-blue-50' },
  { icon: Target, label: 'Medical Image Analysis', color: 'text-emerald-600', bg: 'bg-emerald-50' },
  { icon: Users, label: 'Clinical Decision Support', color: 'text-violet-600', bg: 'bg-violet-50' },
  { icon: FlaskConical, label: 'Research & Education', color: 'text-orange-600', bg: 'bg-orange-50' },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 bg-surface relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left illustration */}
          <div className="relative">
            <div className="w-full max-w-md mx-auto">
              <div className="relative rounded-3xl bg-gradient-to-br from-primary to-accent p-1">
                <div className="rounded-3xl bg-white p-8">
                  <div className="space-y-6">
                    <div className="flex items-center gap-4">
                      <div className="w-14 h-14 rounded-2xl bg-primary/10 flex items-center justify-center">
                        <FlaskConical className="w-7 h-7 text-primary" />
                      </div>
                      <div>
                        <h4 className="font-bold text-text-primary">MedVQA Research</h4>
                        <p className="text-sm text-text-secondary">Visual Question Answering</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      {[
                        { label: 'Image Feature Extraction', value: 92 },
                        { label: 'Question Understanding', value: 88 },
                        { label: 'Answer Generation', value: 94 },
                        { label: 'Clinical Relevance', value: 90 },
                      ].map((bar) => (
                        <div key={bar.label}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-text-secondary">{bar.label}</span>
                            <span className="font-semibold text-text-primary">{bar.value}%</span>
                          </div>
                          <div className="h-2 rounded-full bg-gray-100">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                              style={{ width: `${bar.value}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right content */}
          <div>
            <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
              About
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-6">
              About This Project
            </h2>
            <p className="text-text-secondary text-lg leading-relaxed mb-6">
              Medical Image Understanding through Visual Question Answering using deep learning
              models. This research project explores the intersection of computer vision and
              natural language processing for medical diagnostics.
            </p>
            <p className="text-text-secondary leading-relaxed mb-8">
              By combining advanced image encoders with language models, we enable intuitive
              interaction with medical imaging data. Clinicians and researchers can ask natural
              language questions about radiological images and receive AI-generated answers
              backed by confidence scores and visual explanations.
            </p>

            {/* Highlight chips */}
            <div className="grid grid-cols-2 gap-4">
              {highlights.map((h) => (
                <div
                  key={h.label}
                  className="flex items-center gap-3 p-3 rounded-xl bg-white border border-gray-100 shadow-sm"
                >
                  <div className={`w-10 h-10 rounded-lg ${h.bg} flex items-center justify-center flex-shrink-0`}>
                    <h.icon className={`w-5 h-5 ${h.color}`} />
                  </div>
                  <span className="text-sm font-medium text-text-primary">{h.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
