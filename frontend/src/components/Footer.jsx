import { Activity, Github, Mail, ExternalLink } from 'lucide-react';

const footerLinks = [
  { label: 'Home', href: '#home' },
  { label: 'Analysis', href: '#analysis' },
  { label: 'Models', href: '#models' },
  { label: 'Dataset', href: '#dataset' },
  { label: 'History', href: '#history' },
  { label: 'About', href: '#about' },
];

export default function Footer() {
  return (
    <footer className="bg-gray-900 text-white relative overflow-hidden">
      {/* Decorative gradient */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid md:grid-cols-3 gap-12 mb-12">
          {/* Brand */}
          <div>
            <a href="#home" className="flex items-center gap-2 group mb-5">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-bold">
                Med<span className="text-accent">VQA</span>
              </span>
            </a>
            <p className="text-gray-400 text-sm leading-relaxed max-w-xs">
              Medical Visual Question Answering System — AI-powered diagnostic analysis
              for medical imaging research and education.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-5">
              Quick Links
            </h4>
            <ul className="space-y-3">
              {footerLinks.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    className="text-gray-400 hover:text-white text-sm transition-colors inline-flex items-center gap-1"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-5">
              Resources
            </h4>
            <ul className="space-y-3">
              <li>
                <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors inline-flex items-center gap-2">
                  <Github className="w-4 h-4" /> GitHub Repository
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors inline-flex items-center gap-2">
                  <ExternalLink className="w-4 h-4" /> Research Paper
                </a>
              </li>
              <li>
                <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors inline-flex items-center gap-2">
                  <Mail className="w-4 h-4" /> Contact Us
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-8 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            © {new Date().getFullYear()} MedVQA. Built for research and educational purposes.
          </p>
          <p className="text-gray-500 text-sm">
            Powered by Deep Learning & Computer Vision
          </p>
        </div>
      </div>
    </footer>
  );
}
