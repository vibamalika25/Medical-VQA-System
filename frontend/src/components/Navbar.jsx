import { useState, useEffect } from 'react';
import { Menu, X, Activity } from 'lucide-react';

const navLinks = [
  { label: 'Home', href: '#home' },
  { label: 'Analysis', href: '#analysis' },
  { label: 'Models', href: '#models' },
  { label: 'Dataset', href: '#dataset' },
  { label: 'About', href: '#about' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 30);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? 'bg-white/95 backdrop-blur-xl shadow-[0_1px_3px_rgba(0,0,0,0.08)] border-b border-border'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="flex items-center justify-between h-18 lg:h-20">
          {/* Logo */}
          <a href="#home" className="flex items-center gap-3 group">
            <div className={`w-9 h-9 rounded-lg flex items-center justify-center transition-all duration-300 ${
              scrolled ? 'bg-primary shadow-md shadow-primary/20' : 'bg-white/15 backdrop-blur-sm'
            }`}>
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div className="flex flex-col">
              <span className={`text-[15px] font-bold tracking-tight leading-none transition-colors ${
                scrolled ? 'text-text-primary' : 'text-white'
              }`}>
                MedVQA
              </span>
              <span className={`text-[9px] font-medium uppercase tracking-[0.15em] leading-none mt-0.5 transition-colors ${
                scrolled ? 'text-text-muted' : 'text-white/50'
              }`}>
                Medical AI Platform
              </span>
            </div>
          </a>

          {/* Desktop links */}
          <div className="hidden lg:flex items-center gap-0.5">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className={`px-4 py-2 rounded-lg text-[13px] font-medium transition-all duration-200 ${
                  scrolled
                    ? 'text-text-secondary hover:text-primary hover:bg-primary/5'
                    : 'text-white/70 hover:text-white hover:bg-white/10'
                }`}
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* CTA */}
          <a
            href="#analysis"
            className={`hidden lg:inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-[13px] font-semibold transition-all duration-300 ${
              scrolled
                ? 'bg-primary text-white shadow-md shadow-primary/20 hover:shadow-lg hover:shadow-primary/30 hover:-translate-y-px'
                : 'bg-white text-primary-dark hover:bg-white/90'
            }`}
          >
            Launch Demo
          </a>

          {/* Mobile toggle */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className={`lg:hidden p-2 rounded-lg transition-colors ${
              scrolled ? 'text-text-primary hover:bg-surface' : 'text-white hover:bg-white/10'
            }`}
          >
            {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="lg:hidden bg-white border-t border-border shadow-2xl animate-fade-in">
          <div className="px-6 py-5 space-y-1">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                className="block px-4 py-2.5 rounded-lg text-text-secondary hover:text-primary hover:bg-primary/5 text-sm font-medium transition-colors"
              >
                {link.label}
              </a>
            ))}
            <div className="pt-3">
              <a
                href="#analysis"
                onClick={() => setMobileOpen(false)}
                className="block text-center px-4 py-3 rounded-lg bg-primary text-white text-sm font-semibold shadow-lg shadow-primary/20"
              >
                Launch Demo
              </a>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
