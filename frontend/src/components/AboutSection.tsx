import { Brain, Target, Shield, Zap, BarChart3, CheckCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';

export function AboutSection() {
  const features = [
    {
      icon: Brain,
      title: 'Multi-Scale Attention',
      description: 'Advanced neural architecture with attention mechanisms that focus on relevant brain regions across multiple scales for enhanced tumor detection accuracy.'
    },
    {
      icon: Target,
      title: 'Precise Detection',
      description: 'State-of-the-art deep learning models trained on diverse MRI datasets to identify brain tumors with exceptional sensitivity and specificity.'
    },
    {
      icon: Shield,
      title: 'Uncertainty Quantification',
      description: 'Bayesian neural networks provide confidence intervals and uncertainty estimates, crucial for clinical decision-making processes.'
    },
    {
      icon: Zap,
      title: 'Real-Time Analysis',
      description: 'Optimized inference pipeline delivers rapid analysis results while maintaining high accuracy standards for clinical workflows.'
    }
  ];

  const metrics = [
    { label: 'Accuracy', value: '96.8%', color: 'text-green-600' },
    { label: 'Sensitivity', value: '94.2%', color: 'text-blue-600' },
    { label: 'Specificity', value: '97.5%', color: 'text-purple-600' },
    { label: 'Processing Time', value: '<3s', color: 'text-orange-600' }
  ];

  const technicalSpecs = [
    'Convolutional Neural Networks with Attention Mechanisms',
    'Bayesian Deep Learning for Uncertainty Quantification',
    'Multi-Scale Feature Extraction',
    'DICOM Standard Compatibility',
    'Clinical Risk Stratification Algorithms',
    'FDA Guidelines Compliant Development Process'
  ];

  return (
    <section id="about" className="min-h-screen py-20 px-6 bg-background/30 backdrop-blur-[1px]">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16 fade-in">
          <h2 className="text-medical-lg mb-6">Advanced AI for Medical Imaging</h2>
          <p className="text-xl text-muted-foreground max-w-4xl mx-auto">
            NeuroVision AI represents a breakthrough in medical imaging analysis, combining cutting-edge 
            deep learning with clinical expertise to assist healthcare professionals in brain tumor detection.
          </p>
        </div>

        {/* Key Features */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="card-medical p-6 text-center">
                <div className="w-16 h-16 bg-gold-gradient rounded-xl flex items-center justify-center mx-auto mb-4">
                  <Icon className="w-8 h-8 text-primary-foreground" />
                </div>
                <h3 className="text-lg font-semibold mb-3">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </Card>
            );
          })}
        </div>

        {/* Performance Metrics */}
        <div className="mb-16">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold mb-4">Performance Metrics</h3>
            <p className="text-muted-foreground">
              Validated on comprehensive datasets with clinical ground truth
            </p>
          </div>
          
          <div className="grid md:grid-cols-4 gap-6">
            {metrics.map((metric, index) => (
              <Card key={index} className="card-medical p-6 text-center">
                <BarChart3 className="w-8 h-8 text-primary mx-auto mb-3" />
                <div className={`text-3xl font-bold ${metric.color} mb-2`}>
                  {metric.value}
                </div>
                <div className="text-sm text-muted-foreground font-medium">
                  {metric.label}
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Technical Specifications */}
        <Card className="card-medical p-8">
          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-2xl font-bold mb-6 flex items-center">
                <Brain className="w-8 h-8 text-primary mr-3" />
                Technical Architecture
              </h3>
              <div className="space-y-3">
                {technicalSpecs.map((spec, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                    <span className="text-foreground/90">{spec}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="lg:pl-8">
              <h3 className="text-2xl font-bold mb-6">Clinical Integration</h3>
              <div className="space-y-4">
                <div className="p-4 bg-accent/50 rounded-lg">
                  <h4 className="font-semibold mb-2">Risk Stratification</h4>
                  <p className="text-sm text-muted-foreground">
                    Automated classification into low, medium, and high-risk categories 
                    based on imaging features and clinical protocols.
                  </p>
                </div>
                
                <div className="p-4 bg-accent/50 rounded-lg">
                  <h4 className="font-semibold mb-2">Uncertainty Awareness</h4>
                  <p className="text-sm text-muted-foreground">
                    Bayesian inference provides confidence intervals, helping clinicians 
                    understand model limitations and make informed decisions.
                  </p>
                </div>
                
                <div className="p-4 bg-accent/50 rounded-lg">
                  <h4 className="font-semibold mb-2">Clinical Recommendations</h4>
                  <p className="text-sm text-muted-foreground">
                    Evidence-based recommendations generated based on imaging findings 
                    and established clinical guidelines.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
}