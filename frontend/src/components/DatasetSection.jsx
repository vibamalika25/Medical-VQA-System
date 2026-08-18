import { Database, Image, MessageCircleQuestion, BarChart3 } from 'lucide-react';

const stats = [
  { icon: Image, value: '315', label: 'Medical Images', color: 'text-blue-600', bg: 'bg-blue-50' },
  { icon: MessageCircleQuestion, value: '3,515', label: 'Question-Answer Pairs', color: 'text-emerald-600', bg: 'bg-emerald-50' },
  { icon: Database, value: '11', label: 'Body Regions Covered', color: 'text-violet-600', bg: 'bg-violet-50' },
  { icon: BarChart3, value: '4', label: 'Imaging Modalities', color: 'text-orange-600', bg: 'bg-orange-50' },
];

export default function DatasetSection() {
  return (
    <section id="dataset" className="py-24 bg-surface relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left content */}
          <div>
            <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
              Dataset
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-6">
              VQA-RAD Radiology Dataset
            </h2>
            <p className="text-text-secondary text-lg leading-relaxed mb-6">
              Our system is trained on the <strong className="text-text-primary">VQA-RAD</strong> (Visual Question Answering in Radiology)
              dataset, a curated collection of clinician-generated questions and answers paired with radiology images.
            </p>
            <p className="text-text-secondary leading-relaxed mb-8">
              The dataset covers multiple imaging modalities including X-ray, CT, and MRI scans across
              various body parts. Questions range from identifying abnormalities and characterizing
              findings to determining imaging modality and anatomical location. This diverse dataset
              enables our models to generalize across different clinical scenarios.
            </p>

            <div className="flex flex-wrap gap-3">
              {['X-ray', 'CT Scan', 'MRI', 'Head', 'Chest', 'Abdomen'].map((tag) => (
                <span
                  key={tag}
                  className="px-4 py-2 rounded-xl bg-white border border-gray-200 text-sm font-medium text-text-secondary hover:border-primary hover:text-primary transition-colors"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Right stats grid */}
          <div className="grid grid-cols-2 gap-5">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 hover:shadow-lg hover:-translate-y-1 transition-all duration-300"
              >
                <div className={`w-12 h-12 rounded-xl ${stat.bg} flex items-center justify-center mb-4`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                <p className="text-3xl font-bold text-text-primary">{stat.value}</p>
                <p className="text-text-secondary text-sm mt-1">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
