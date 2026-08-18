import { Brain, Network, BookOpenText, Cpu } from 'lucide-react';

const models = [
  {
    name: 'CNN + LSTM',
    icon: Network,
    accuracy: 82,
    description:
      'Combines Convolutional Neural Networks for image feature extraction with Long Short-Term Memory networks for sequential question understanding and answer generation.',
    color: 'from-blue-600 to-cyan-500',
    bg: 'bg-blue-50',
  },
  {
    name: 'BioBERT',
    icon: BookOpenText,
    accuracy: 87,
    description:
      'Pre-trained biomedical language representation model fine-tuned on medical literature. Excels at understanding clinical terminology and medical question intent.',
    color: 'from-emerald-600 to-teal-500',
    bg: 'bg-emerald-50',
  },
  {
    name: 'LXMERT',
    icon: Cpu,
    accuracy: 91,
    description:
      'Learning Cross-Modality Encoder Representations from Transformers. A powerful model that learns joint vision-language representations through cross-modal attention.',
    color: 'from-violet-600 to-purple-500',
    bg: 'bg-violet-50',
  },
  {
    name: 'BLIP-2',
    icon: Brain,
    accuracy: 94,
    description:
      'Bootstrapping Language-Image Pre-training with frozen image encoders and LLMs. Achieves state-of-the-art performance on medical VQA tasks with efficient training.',
    color: 'from-orange-600 to-amber-500',
    bg: 'bg-orange-50',
  },
];

export default function ModelsSection() {
  return (
    <section id="models" className="py-24 bg-white relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
            Models
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
            AI Models Under the Hood
          </h2>
          <p className="text-text-secondary text-lg">
            We leverage multiple state-of-the-art deep learning architectures for robust medical image understanding.
          </p>
        </div>

        {/* Cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {models.map((model) => (
            <div
              key={model.name}
              className="group relative bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden hover:shadow-xl hover:-translate-y-2 transition-all duration-300"
            >
              {/* Gradient header */}
              <div className={`h-2 bg-gradient-to-r ${model.color}`} />

              <div className="p-6">
                <div className={`w-12 h-12 rounded-xl ${model.bg} flex items-center justify-center mb-4`}>
                  <model.icon className="w-6 h-6 text-primary" />
                </div>

                <h3 className="text-lg font-bold text-text-primary mb-2">{model.name}</h3>
                <p className="text-text-secondary text-sm leading-relaxed mb-5">
                  {model.description}
                </p>

                {/* Accuracy bar */}
                <div>
                  <div className="flex items-center justify-between text-sm mb-2">
                    <span className="text-text-secondary">Accuracy</span>
                    <span className="font-bold text-text-primary">{model.accuracy}%</span>
                  </div>
                  <div className="h-2.5 rounded-full bg-gray-100 overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${model.color} transition-all duration-700`}
                      style={{ width: `${model.accuracy}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
