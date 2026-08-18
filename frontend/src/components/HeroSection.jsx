import { ArrowRight, Brain, ShieldCheck, Cpu, BarChart3 } from 'lucide-react';

export default function HeroSection() {
  return (
    <section id="home" className="relative min-h-screen flex items-center hero-gradient overflow-hidden">
      {/* Background decorations */}
      <div className="absolute inset-0">
        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.04]" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }} />
        {/* Gradient orbs */}
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-white/[0.03] rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-32 w-[500px] h-[500px] bg-accent/[0.08] rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-white/[0.02] rounded-full blur-3xl animate-spin-slow" />
      </div>

      <div className="relative max-w-7xl mx-auto px-6 lg:px-8 w-full py-32 lg:py-0">
        <div className="grid lg:grid-cols-2 gap-16 lg:gap-20 items-center">
          {/* Content */}
          <div className="text-white">
            <div className="inline-flex items-center gap-2.5 px-4 py-2 rounded-full bg-white/[0.08] backdrop-blur-sm border border-white/[0.12] text-[13px] font-medium mb-10">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
              </span>
              AI-Powered Diagnostic Platform
            </div>

            <h1 className="text-[42px] sm:text-5xl lg:text-[56px] font-extrabold leading-[1.1] tracking-tight mb-7">
              Medical Visual
              <br />
              <span className="bg-gradient-to-r from-accent-light via-white to-accent-light bg-clip-text text-transparent animate-gradient">
                Question Answering
              </span>
              <br />
              System
            </h1>

            <p className="text-[17px] text-white/60 max-w-lg mb-10 leading-relaxed">
              Upload radiology images — X-rays, MRIs, CT scans — and query them using
              natural language. Our deep learning models analyze the image and deliver
              diagnostic answers with confidence scoring.
            </p>

            <div className="flex flex-wrap gap-4 mb-16">
              <a
                href="#analysis"
                className="group inline-flex items-center gap-2.5 px-7 py-3.5 rounded-xl bg-white text-primary-dark font-semibold text-[15px] shadow-2xl shadow-black/20 hover:shadow-3xl hover:-translate-y-0.5 transition-all duration-300"
              >
                Try the Demo
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </a>
              <a
                href="#about"
                className="inline-flex items-center gap-2 px-7 py-3.5 rounded-xl border border-white/20 text-white font-medium text-[15px] hover:bg-white/[0.08] transition-all duration-300"
              >
                Learn More
              </a>
            </div>

            {/* Stats row */}
            <div className="flex gap-10 pt-8 border-t border-white/[0.08]">
              {[
                { value: '315+', label: 'Medical Images' },
                { value: '3.5k+', label: 'Q&A Pairs' },
                { value: '94%', label: 'Accuracy' },
              ].map((stat) => (
                <div key={stat.label}>
                  <p className="text-2xl font-bold tracking-tight">{stat.value}</p>
                  <p className="text-[12px] text-white/40 mt-1 uppercase tracking-wider font-medium">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right side — Enterprise illustration */}
          <div className="hidden lg:block">
            <div className="relative">
              {/* Main card */}
              <div className="relative w-full max-w-[420px] mx-auto">
                <div className="rounded-2xl bg-white/[0.07] backdrop-blur-xl border border-white/[0.1] p-8 animate-float">
                  {/* Header */}
                  <div className="flex items-center gap-3 mb-8">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-accent to-primary flex items-center justify-center">
                      <Brain className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-white font-semibold text-[15px]">AI Diagnostic Engine</h3>
                      <p className="text-white/40 text-xs">Multimodal Analysis</p>
                    </div>
                  </div>

                  {/* Analysis bars */}
                  <div className="space-y-4">
                    {[
                      { label: 'Image Feature Extraction', value: 96 },
                      { label: 'NLP Question Parsing', value: 92 },
                      { label: 'Answer Generation', value: 94 },
                    ].map((bar) => (
                      <div key={bar.label}>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-white/50">{bar.label}</span>
                          <span className="text-white font-semibold">{bar.value}%</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-white/[0.08]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-accent to-primary-light"
                            style={{ width: `${bar.value}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Floating badges */}
                <div className="absolute -top-5 -right-5 glass-card rounded-xl px-4 py-3 shadow-2xl flex items-center gap-3 animate-pulse-ring">
                  <div className="w-9 h-9 rounded-lg bg-success/10 flex items-center justify-center">
                    <ShieldCheck className="w-5 h-5 text-success" />
                  </div>
                  <div>
                    <p className="text-[10px] text-text-muted uppercase tracking-wider font-medium">Status</p>
                    <p className="text-[13px] font-bold text-success">Operational</p>
                  </div>
                </div>

                <div className="absolute -bottom-5 -left-5 glass-card rounded-xl px-4 py-3 shadow-2xl flex items-center gap-3" style={{ animation: 'pulse-ring 3s ease-in-out infinite 1.5s' }}>
                  <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Cpu className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <p className="text-[10px] text-text-muted uppercase tracking-wider font-medium">Models</p>
                    <p className="text-[13px] font-bold text-primary">4 Active</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
