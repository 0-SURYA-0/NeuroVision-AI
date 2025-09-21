import { ArrowDown, Brain, Zap, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import brainBackground from '@/assets/brain-background.jpg';

interface HeroSectionProps {
  onScrollToUpload: () => void;
}

export function HeroSection({ onScrollToUpload }: HeroSectionProps) {
  return (
    <section 
      id="hero" 
      className="min-h-screen flex items-center justify-center relative overflow-hidden"
      style={{
        backgroundImage: `url(${brainBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed'
      }}
    >
      {/* Medical Overlay - More Translucent */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-background/20 to-background/40" />
      
      {/* Content */}
      <div className="relative z-10 text-center max-w-6xl mx-auto px-6">
        <div className="fade-in">
          {/* Main Title */}
          <h1 className="text-medical-xl mb-6">
            NeuroVision AI
          </h1>
          
          {/* Subtitle */}
          <p className="text-medical-lg mb-8 max-w-4xl mx-auto">
            Multi-Scale Attention-Guided MRI Brain Tumor Detection System
            <span className="block mt-2 text-xl text-muted-foreground">
              with Bayesian Uncertainty Quantification and Clinical Risk Stratification
            </span>
          </p>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12 max-w-4xl mx-auto">
            <div className="card-medical p-6">
              <Brain className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Advanced AI Detection</h3>
              <p className="text-sm text-muted-foreground">
                Deep learning with multi-scale attention mechanisms for precise tumor identification
              </p>
            </div>
            
            <div className="card-medical p-6">
              <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Uncertainty Quantification</h3>
              <p className="text-sm text-muted-foreground">
                Bayesian neural networks provide confidence estimates for clinical decision-making
              </p>
            </div>
            
            <div className="card-medical p-6">
              <Zap className="w-12 h-12 text-primary mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Clinical Integration</h3>
              <p className="text-sm text-muted-foreground">
                Risk stratification and recommendations tailored for medical professionals
              </p>
            </div>
          </div>

          {/* CTA Button */}
          <div className="slide-up">
            <Button
              onClick={onScrollToUpload}
              className="btn-medical text-lg group"
            >
              Start Analysis
              <ArrowDown className="ml-2 w-5 h-5 group-hover:animate-bounce" />
            </Button>
          </div>
        </div>
      </div>

      {/* Animated Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-primary rounded-full flex justify-center">
          <div className="w-1 h-3 bg-primary rounded-full mt-2 animate-pulse-gold" />
        </div>
      </div>
    </section>
  );
}