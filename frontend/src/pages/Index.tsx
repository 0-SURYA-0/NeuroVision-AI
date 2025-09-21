import { useState, useEffect } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { HeroSection } from '@/components/HeroSection';
import { ImageUploader } from '@/components/ImageUploader';
import { AboutSection } from '@/components/AboutSection';
import { AuthorsSection } from '@/components/AuthorsSection';
import { Footer } from '@/components/Footer';
import { Toaster } from "@/components/ui/toaster";
import brainBackground from '@/assets/brain-background.jpg';

const Index = () => {
  const [activeSection, setActiveSection] = useState('hero');

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
      setActiveSection(sectionId);
    }
  };

  // Update active section on scroll
  useEffect(() => {
    const handleScroll = () => {
      const sections = ['hero', 'upload', 'about', 'authors'];
      const scrollPosition = window.scrollY + 100;

      for (const section of sections) {
        const element = document.getElementById(section);
        if (element) {
          const { offsetTop, offsetHeight } = element;
          if (scrollPosition >= offsetTop && scrollPosition < offsetTop + offsetHeight) {
            setActiveSection(section);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-transparent flex" 
         style={{
           backgroundImage: `url(${brainBackground})`,
           backgroundSize: 'cover',
           backgroundPosition: 'center',
           backgroundAttachment: 'fixed'
         }}>
      {/* Sidebar Navigation */}
      <Sidebar 
        activeSection={activeSection} 
        onSectionChange={scrollToSection}
      />

      {/* Main Content */}
      <div className="flex-1 w-full">
        <HeroSection onScrollToUpload={() => scrollToSection('upload')} />
        <ImageUploader />
        <AboutSection />
        <AuthorsSection />
        <Footer />
      </div>

      <Toaster />
    </div>
  );
};

export default Index;
