import { Brain, Mail, MapPin, Phone, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-background/40 backdrop-blur-[2px] border-t border-border/30 py-16 px-6">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gold-gradient rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h3 className="text-xl font-bold">NeuroVision AI</h3>
                <p className="text-sm text-muted-foreground">Medical Imaging AI</p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Advanced AI-powered brain tumor detection system with Bayesian uncertainty 
              quantification for clinical decision support.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <div className="space-y-2">
              {[
                { label: 'About Project', href: '#about' },
                { label: 'AI Analysis', href: '#upload' },
                { label: 'Research Team', href: '#authors' },
                { label: 'Technical Docs', href: '#' }
              ].map((link, index) => (
                <div key={index}>
                  <a 
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {link.label}
                  </a>
                </div>
              ))}
            </div>
          </div>

          {/* Research */}
          <div>
            <h4 className="font-semibold mb-4">Research</h4>
            <div className="space-y-2">
              {[
                { label: 'Publications', href: '#' },
                { label: 'Dataset', href: '#' },
                { label: 'Methodology', href: '#' },
                { label: 'Collaboration', href: '#' }
              ].map((link, index) => (
                <div key={index}>
                  <a 
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors flex items-center"
                  >
                    {link.label}
                    <ExternalLink className="w-3 h-3 ml-1" />
                  </a>
                </div>
              ))}
            </div>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <div className="space-y-3">
              <div className="flex items-start space-x-2">
                <Mail className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="text-muted-foreground">research@neurovision-ai.org</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-2">
                <MapPin className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="text-muted-foreground">
                    Multi-institutional Research<br />
                    Stanford, Johns Hopkins, MIT, UCSF
                  </p>
                </div>
              </div>
              
              <div className="flex items-start space-x-2">
                <Phone className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="text-muted-foreground">+1 (555) 123-4567</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-border/50 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-sm text-muted-foreground">
              © {currentYear} NeuroVision AI Research Consortium. All rights reserved.
            </div>
            
            <div className="flex space-x-6 text-sm">
              <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                Privacy Policy
              </a>
              <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                Terms of Use
              </a>
              <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                Research Ethics
              </a>
            </div>
          </div>
          
          {/* Disclaimer */}
          <div className="mt-6 p-4 bg-accent/30 rounded-lg">
            <p className="text-xs text-muted-foreground text-center leading-relaxed">
              <strong>Medical Disclaimer:</strong> NeuroVision AI is a research prototype designed for 
              investigational purposes only. This system is not FDA-approved for clinical diagnosis 
              and should not replace professional medical judgment. Always consult qualified healthcare 
              professionals for medical decisions.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}