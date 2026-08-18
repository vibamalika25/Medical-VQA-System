import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import FeaturesSection from './components/FeaturesSection';
import AnalysisSection from './components/AnalysisSection';
import ModelsSection from './components/ModelsSection';
import DatasetSection from './components/DatasetSection';
import AboutSection from './components/AboutSection';
import Footer from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <AnalysisSection />
      <ModelsSection />
      <DatasetSection />
      <AboutSection />
      <Footer />
    </div>
  );
}
