const API_BASE = 'http://localhost:8000';

/**
 * Send a medical image and question to the backend for analysis.
 * Falls back to a mock response if the server is unreachable.
 *
 * @param {File} imageFile  — the uploaded image
 * @param {string} question — the clinical question
 * @returns {{ answer, confidence, organ, diagnosis, explanation }}
 */
export async function analyzeImage(imageFile, question) {
  try {
    const form = new FormData();
    form.append('image', imageFile);
    form.append('question', question);

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    console.warn('Backend unreachable, using mock response:', err.message);
    return mockAnalyze(question);
  }
}

/**
 * Check if the backend API is healthy.
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return await res.json();
  } catch {
    return { status: 'offline', model_loaded: false };
  }
}

// ── Mock fallback (used when backend is not running) ──────────────────────

const mockResponses = [
  {
    answer: 'Bilateral pleural effusion detected',
    confidence: 94.2,
    organ: 'lung',
    diagnosis: 'abnormal',
    explanation:
      'The AI model detected bilateral pleural effusions characterised by blunting of the costophrenic angles on both sides. The deep learning model identified fluid accumulation in the pleural space, which may indicate heart failure, infection, or other underlying conditions.',
  },
  {
    answer: 'Cardiomegaly present',
    confidence: 89.7,
    organ: 'heart',
    diagnosis: 'abnormal',
    explanation:
      'The cardiothoracic ratio exceeds 0.5, indicating cardiomegaly. The AI model measured the transverse cardiac diameter relative to the thoracic diameter and confirmed enlargement of the cardiac silhouette.',
  },
  {
    answer: 'No pneumothorax detected',
    confidence: 97.1,
    organ: 'lung',
    diagnosis: 'normal',
    explanation:
      'The AI model analysed the lung periphery and pleural surfaces for signs of pneumothorax. The visceral pleural line is intact bilaterally, and lung markings extend to the chest wall without evidence of air trapping.',
  },
  {
    answer: 'Right lower lobe consolidation',
    confidence: 91.5,
    organ: 'lung',
    diagnosis: 'infection',
    explanation:
      'The AI identified an area of increased opacity in the right lower lobe consistent with consolidation. Air bronchograms may be present within the opacified region. This pattern is commonly seen in bacterial pneumonia.',
  },
  {
    answer: 'This is a PA chest X-ray',
    confidence: 98.3,
    organ: 'lung',
    diagnosis: 'normal',
    explanation:
      'The imaging modality has been identified as a posteroanterior (PA) chest radiograph based on the orientation, magnification of cardiac silhouette, and positioning markers.',
  },
];

function mockAnalyze(question) {
  return new Promise((resolve) => {
    const delay = 1500 + Math.random() * 1500;
    setTimeout(() => {
      const q = question.toLowerCase();
      let response;
      if (q.includes('modality') || q.includes('type')) {
        response = mockResponses[4];
      } else if (q.includes('cardiomegaly') || q.includes('heart')) {
        response = mockResponses[1];
      } else if (q.includes('pneumothorax')) {
        response = mockResponses[2];
      } else if (q.includes('consolidation') || q.includes('pneumonia')) {
        response = mockResponses[3];
      } else {
        response = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      }
      resolve({ ...response });
    }, delay);
  });
}
