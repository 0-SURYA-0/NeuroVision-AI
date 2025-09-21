import React, { useState, useRef } from 'react';
import { Upload, FileImage, AlertCircle, Check, Brain } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface UploadResult {
  prediction: 'tumor' | 'no-tumor';
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  uncertainty: number;
  recommendations: string[];
}

export function ImageUploader() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
  
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelection(files[0]);
    }
  };

  const handleFileSelection = (file: File) => {
    if (!acceptedTypes.includes(file.type)) {
      setError('Please select a PNG, JPG, or JPEG file.');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB.');
      return;
    }

    setError(null);
    setSelectedFile(file);
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      // Simulate AI analysis
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock result
      const mockResult: UploadResult = {
        prediction: Math.random() > 0.7 ? 'tumor' : 'no-tumor',
        confidence: Math.random() * 0.3 + 0.7, // 70-100%
        riskLevel: Math.random() > 0.6 ? 'low' : Math.random() > 0.3 ? 'medium' : 'high',
        uncertainty: Math.random() * 0.2 + 0.1, // 10-30%
        recommendations: [
          'Further evaluation recommended',
          'Consider additional imaging studies',
          'Consult with neurologist or neurosurgeon'
        ]
      };

      setResult(mockResult);
    } catch (err) {
      setError('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600';
      case 'medium': return 'text-yellow-600';  
      case 'high': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <section id="upload" className="min-h-screen py-20 px-6 bg-background/20 backdrop-blur-[1px]">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12 fade-in">
          <h2 className="text-medical-lg mb-4">AI-Powered MRI Analysis</h2>
          <p className="text-lg text-muted-foreground">
            Upload your MRI brain scan for advanced tumor detection analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <Card className="bg-card/40 backdrop-blur-[2px] border-border/40 rounded-xl shadow-medical p-8">
            <div className="space-y-6">
              <div className="text-center">
                <Brain className="w-16 h-16 text-primary mx-auto mb-4" />
                <h3 className="text-2xl font-semibold mb-2">Upload MRI Scan</h3>
                <p className="text-muted-foreground">
                  DICOM, PNG, or JPG formats accepted
                </p>
              </div>

              {/* Drag & Drop Area */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`
                  border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer
                  ${isDragOver 
                    ? 'border-primary bg-primary/5 scale-105' 
                    : 'border-border hover:border-primary/50'
                  }
                `}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragOver ? 'text-primary' : 'text-muted-foreground'}`} />
                <div className="space-y-2">
                  <p className="text-lg font-medium">
                    {isDragOver ? 'Drop file here' : 'Drag & drop your MRI scan'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                </div>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,.dcm"
                onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
                className="hidden"
              />

              {/* Selected File */}
              {selectedFile && (
                <div className="flex items-center space-x-3 p-4 bg-accent rounded-lg">
                  <FileImage className="w-8 h-8 text-primary" />
                  <div className="flex-1">
                    <p className="font-medium">{selectedFile.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <Check className="w-6 h-6 text-green-600" />
                </div>
              )}

              {/* Error Alert */}
              {error && (
                <Alert className="border-destructive">
                  <AlertCircle className="w-4 h-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Analyze Button */}
              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
                className="w-full btn-medical"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-primary-foreground/20 border-t-primary-foreground rounded-full animate-spin mr-2" />
                    Analyzing MRI...
                  </>
                ) : (
                  'Analyze MRI Scan'
                )}
              </Button>
            </div>
          </Card>

          {/* Results Section */}
          <Card className={`bg-card/40 backdrop-blur-[2px] border-border/40 rounded-xl shadow-medical p-8 ${result ? 'slide-up' : ''}`}>
            {result ? (
              <div className="space-y-6">
                <div className="text-center">
                  <div className={`w-20 h-20 rounded-full mx-auto mb-4 flex items-center justify-center ${
                    result.prediction === 'tumor' ? 'bg-red-100' : 'bg-green-100'
                  }`}>
                    <Brain className={`w-10 h-10 ${
                      result.prediction === 'tumor' ? 'text-red-600' : 'text-green-600'
                    }`} />
                  </div>
                  <h3 className="text-2xl font-bold mb-2">
                    {result.prediction === 'tumor' ? 'Tumor Detected' : 'No Tumor Detected'}
                  </h3>
                </div>

                {/* Confidence & Risk */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-accent rounded-lg">
                    <div className="text-2xl font-bold text-primary">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-muted-foreground">Confidence</div>
                  </div>
                  <div className="text-center p-4 bg-accent rounded-lg">
                    <div className={`text-2xl font-bold capitalize ${getRiskColor(result.riskLevel)}`}>
                      {result.riskLevel}
                    </div>
                    <div className="text-sm text-muted-foreground">Risk Level</div>
                  </div>
                </div>

                {/* Uncertainty */}
                <div className="p-4 bg-accent rounded-lg">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium">Uncertainty</span>
                    <span className="text-sm">{(result.uncertainty * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary h-2 rounded-full"
                      style={{ width: `${result.uncertainty * 100}%` }}
                    />
                  </div>
                </div>

                {/* Recommendations */}
                <div>
                  <h4 className="font-semibold mb-3">Clinical Recommendations</h4>
                  <div className="space-y-2">
                    {result.recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-2">
                        <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                  <Brain className="w-10 h-10 text-muted-foreground" />
                </div>
                <h3 className="text-xl font-semibold text-muted-foreground mb-2">
                  Analysis Results
                </h3>
                <p className="text-muted-foreground">
                  Upload and analyze an MRI scan to see results here
                </p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </section>
  );
}