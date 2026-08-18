import { History, ImageIcon, HelpCircle, MessageSquare, TrendingUp } from 'lucide-react';

const historyData = [
  {
    id: 1,
    imageType: 'Chest X-ray',
    question: 'What abnormality is present in the lungs?',
    answer: 'Bilateral pleural effusion detected',
    confidence: 94.2,
  },
  {
    id: 2,
    imageType: 'Head CT',
    question: 'Is there any mass present?',
    answer: 'No intracranial mass identified',
    confidence: 91.8,
  },
  {
    id: 3,
    imageType: 'Chest X-ray',
    question: 'Is cardiomegaly present?',
    answer: 'Yes, cardiomegaly is present',
    confidence: 89.7,
  },
  {
    id: 4,
    imageType: 'Abdominal MRI',
    question: 'What organ is abnormal?',
    answer: 'Hepatomegaly detected in liver',
    confidence: 86.3,
  },
  {
    id: 5,
    imageType: 'Chest X-ray',
    question: 'Is the trachea midline?',
    answer: 'Yes, trachea is midline',
    confidence: 97.1,
  },
];

function ConfidenceBadge({ value }) {
  let color = 'bg-red-100 text-red-700';
  if (value >= 90) color = 'bg-green-100 text-green-700';
  else if (value >= 80) color = 'bg-blue-100 text-blue-700';
  else if (value >= 70) color = 'bg-yellow-100 text-yellow-700';

  return (
    <span className={`px-3 py-1 rounded-full text-xs font-bold ${color}`}>
      {value}%
    </span>
  );
}

export default function HistorySection() {
  return (
    <section id="history" className="py-24 bg-white relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center max-w-2xl mx-auto mb-14">
          <span className="inline-block px-4 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-semibold mb-4">
            History
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-text-primary mb-4">
            Recent Analysis History
          </h2>
          <p className="text-text-secondary text-lg">
            View past queries and their diagnostic predictions.
          </p>
        </div>

        {/* Table card */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm overflow-hidden">
          {/* Desktop table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-surface border-b border-gray-100">
                  <th className="px-6 py-4 text-xs font-semibold text-text-secondary uppercase tracking-wider">
                    <span className="flex items-center gap-2"><ImageIcon className="w-4 h-4" /> Image</span>
                  </th>
                  <th className="px-6 py-4 text-xs font-semibold text-text-secondary uppercase tracking-wider">
                    <span className="flex items-center gap-2"><HelpCircle className="w-4 h-4" /> Question</span>
                  </th>
                  <th className="px-6 py-4 text-xs font-semibold text-text-secondary uppercase tracking-wider">
                    <span className="flex items-center gap-2"><MessageSquare className="w-4 h-4" /> Predicted Answer</span>
                  </th>
                  <th className="px-6 py-4 text-xs font-semibold text-text-secondary uppercase tracking-wider">
                    <span className="flex items-center gap-2"><TrendingUp className="w-4 h-4" /> Confidence</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {historyData.map((row, i) => (
                  <tr
                    key={row.id}
                    className={`border-b border-gray-50 hover:bg-primary/[0.02] transition-colors ${
                      i === historyData.length - 1 ? 'border-b-0' : ''
                    }`}
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                          <ImageIcon className="w-5 h-5 text-primary" />
                        </div>
                        <span className="text-sm font-medium text-text-primary">{row.imageType}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm text-text-secondary max-w-xs">{row.question}</td>
                    <td className="px-6 py-4 text-sm font-medium text-text-primary max-w-xs">{row.answer}</td>
                    <td className="px-6 py-4">
                      <ConfidenceBadge value={row.confidence} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile card view */}
          <div className="md:hidden divide-y divide-gray-100">
            {historyData.map((row) => (
              <div key={row.id} className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                      <ImageIcon className="w-4 h-4 text-primary" />
                    </div>
                    <span className="text-sm font-medium text-text-primary">{row.imageType}</span>
                  </div>
                  <ConfidenceBadge value={row.confidence} />
                </div>
                <p className="text-sm text-text-secondary"><strong>Q:</strong> {row.question}</p>
                <p className="text-sm text-text-primary font-medium"><strong>A:</strong> {row.answer}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
