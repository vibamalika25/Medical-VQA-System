import { Eye, MessageSquareText, Radiation, Zap } from 'lucide-react';

const features = [
  {
    icon: Eye,
    title: 'Image Understanding',
    description: 'Advanced CNN-based models analyze medical images to extract visual features and detect abnormalities with high precision.',
    color: 'from-blue-500 to-cyan-500',
    bg: 'bg-blue-50',
    iconColor: 'text-blue-600',
  },
  {
    icon: MessageSquareText,
    title: 'Medical Question Answering',
    description: 'Natural language processing allows you to ask clinical questions about medical images and receive accurate, context-aware answers.',
    color: 'from-emerald-500 to-teal-500',
    bg: 'bg-emerald-50',
    iconColor: 'text-emerald-600',
  },
  {
    icon: Radiation,
    title: 'Radiology Image Analysis',
    description: 'Specialized models trained on radiology datasets including X-rays, CT scans, and MRIs for comprehensive diagnostic support.',
    color: 'from-violet-500 to-purple-500',
    bg: 'bg-violet-50',
    iconColor: 'text-violet-600',
  },
  {
    icon: Zap,
    title: 'Real-time AI Predictions',
    description: 'Get instant predictions with confidence scores and visual explanations, enabling faster and more informed clinical decisions.',
    color: 'from-orange-500 to-amber-500',
    bg: 'bg-orange-50',
    iconColor: 'text-orange-600',
  },
];

export default function FeaturesSection() {
  return (
    <section className="py-24 bg-white relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
            Features
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
            Powered by State-of-the-Art AI
          </h2>
          <p className="text-text-secondary text-lg">
            Our platform combines cutting-edge deep learning with medical imaging expertise
            to deliver accurate diagnostic insights.
          </p>
        </div>

        {/* Cards grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              className="group relative p-6 rounded-2xl bg-white border border-gray-100 shadow-sm hover:shadow-xl hover:-translate-y-2 transition-all duration-300"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              {/* Gradient top bar */}
              <div className={`absolute top-0 left-6 right-6 h-1 rounded-b-full bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-100 transition-opacity`} />

              <div className={`w-14 h-14 rounded-2xl ${feature.bg} flex items-center justify-center mb-5 group-hover:scale-110 transition-transform`}>
                <feature.icon className={`w-7 h-7 ${feature.iconColor}`} />
              </div>

              <h3 className="text-lg font-bold text-text-primary mb-2">
                {feature.title}
              </h3>
              <p className="text-text-secondary text-sm leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
