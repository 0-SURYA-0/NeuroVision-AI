import { useState } from 'react';
import { Brain, Upload, FileText, Users, Menu, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const navigationItems = [
    { id: 'hero', label: 'NeuroVision', icon: Brain },
    { id: 'upload', label: 'Analysis', icon: Upload },
    { id: 'about', label: 'About', icon: FileText },
    { id: 'authors', label: 'Team', icon: Users },
  ];

  return (
    <>
      {/* Mobile Overlay */}
      {!isCollapsed && (
        <div 
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setIsCollapsed(true)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        h-screen sticky top-0 z-40 transition-all duration-300
        ${isCollapsed ? 'w-0 overflow-hidden lg:w-20' : 'w-72 lg:w-80'}
        flex-shrink-0
      `}>
        <div className="h-full bg-card/30 backdrop-blur-[2px] border-r border-border/30 shadow-medical">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
              {!isCollapsed && (
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-gold-gradient rounded-xl flex items-center justify-center">
                    <Brain className="w-6 h-6 text-primary-foreground" />
                  </div>
                  <div>
                    <h1 className="text-xl font-bold text-sidebar-foreground">NeuroVision AI</h1>
                    <p className="text-sm text-sidebar-foreground/60">Medical Imaging</p>
                  </div>
                </div>
              )}
              
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsCollapsed(!isCollapsed)}
                className="lg:hidden"
              >
                {isCollapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
              </Button>
            </div>

            {/* Navigation */}
            <nav className="space-y-2">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeSection === item.id;
                
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      onSectionChange(item.id);
                      setIsCollapsed(true);
                    }}
                    className={`
                      w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all
                      ${isActive 
                        ? 'bg-primary text-primary-foreground shadow-gold' 
                        : 'text-foreground hover:bg-accent hover:text-accent-foreground'
                      }
                      ${isCollapsed ? 'justify-center px-2' : ''}
                    `}
                  >
                    <Icon className={`w-5 h-5 ${isActive ? 'animate-pulse-gold' : ''}`} />
                    {!isCollapsed && (
                      <span className="font-medium">{item.label}</span>
                    )}
                  </button>
                );
              })}
            </nav>

            {/* Footer */}
            {!isCollapsed && (
              <div className="absolute bottom-6 left-6 right-6">
                <div className="bg-accent/50 backdrop-blur-sm rounded-lg p-4 text-center border">
                  <p className="text-sm text-muted-foreground">
                    Advanced AI-powered brain tumor detection
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Mobile Trigger */}
      {isCollapsed && (
        <Button
          variant="default"
          size="icon"
          onClick={() => setIsCollapsed(false)}
          className="fixed top-4 left-4 z-40 lg:hidden btn-medical"
        >
          <Menu className="w-5 h-5" />
        </Button>
      )}
    </>
  );
}