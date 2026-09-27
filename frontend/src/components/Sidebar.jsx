import React from 'react';
import {
  Activity,
  LayoutDashboard,
  Radio,
  BarChart3,
  TrendingUp,
  SlidersHorizontal,
  X,
  Server,
  Cpu,
  ShieldCheck,
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'overview', label: 'OVERVIEW', icon: LayoutDashboard },
  { id: 'live-matches', label: 'LIVE MATCHES', icon: Radio },
  { id: 'match-analysis', label: 'MATCH ANALYSIS', icon: BarChart3 },
  { id: 'probability-swings', label: 'PROBABILITY SWINGS', icon: TrendingUp },
  { id: 'what-if-simulator', label: 'WHAT-IF SIMULATOR', icon: SlidersHorizontal },
];

export default function Sidebar({
  activeSection,
  onNavigate,
  isOpen,
  onClose,
  backendConnected = true,
  liveCount = 0,
  isDemo = false,
}) {
  const handleItemClick = (id) => {
    if (onNavigate) {
      onNavigate(id);
    }
    if (onClose) {
      onClose();
    }
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden transition-opacity"
          aria-hidden="true"
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-slate-950 border-r border-slate-800 flex flex-col justify-between transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Top: Brand Header */}
        <div>
          <div className="p-4 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 flex-shrink-0">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-black tracking-wider text-white uppercase leading-none">
                  CRICKET WIN PREDICTOR
                </div>
                <div className="text-[10px] font-mono tracking-widest text-emerald-400 uppercase mt-1 leading-none">
                  ANALYTICS COMMAND CENTER
                </div>
              </div>
            </div>

            {/* Mobile Close Button */}
            <button
              onClick={onClose}
              className="lg:hidden p-1.5 text-slate-400 hover:text-white rounded-md hover:bg-slate-900 transition"
              aria-label="Close sidebar"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Navigation Anchors */}
          <nav className="p-3 space-y-1">
            <div className="px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest text-slate-500">
              NAVIGATION
            </div>
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => handleItemClick(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-bold tracking-wider transition-all text-left ${
                    isActive
                      ? 'bg-slate-900 text-emerald-400 border border-emerald-500/30 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
                    <span>{item.label}</span>
                  </div>
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#10b981]" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom: System Telemetry */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/80">
          <div className="flex items-center gap-1.5 px-2 mb-2 text-[10px] font-mono uppercase tracking-widest text-slate-400">
            <Cpu className="w-3 h-3 text-sky-400" />
            <span>SYSTEM TELEMETRY</span>
          </div>

          <div className="bg-slate-900/90 border border-slate-800/90 rounded-lg p-2.5 space-y-2 text-[11px] font-mono">
            <div className="flex justify-between items-center">
              <span className="text-slate-500">MODEL</span>
              <span className="text-slate-200 font-semibold">Logistic Regression</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">TEST ROC-AUC</span>
              <span className="text-emerald-400 font-bold">0.9109</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">ARCHITECTURE</span>
              <span className="text-slate-300">120-Ball T20</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">LIVE FEED</span>
              <span className="text-slate-300 truncate max-w-[105px]">Cricket Data API</span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-slate-500">FORMAT</span>
              <span className="text-emerald-400 font-semibold">T20 / T20I</span>
            </div>

            <div className="flex justify-between items-center pt-1 border-t border-slate-800/80">
              <span className="text-slate-500 flex items-center gap-1">
                <Server className="w-2.5 h-2.5 text-slate-500" />
                BACKEND
              </span>
              <span className={`font-semibold flex items-center gap-1 ${backendConnected ? 'text-emerald-400' : 'text-rose-400'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${backendConnected ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                :5000
              </span>
            </div>
          </div>

          <div className="mt-2.5 px-2 text-[10px] text-slate-500 flex items-center justify-between">
            <span className="flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-emerald-500" />
              Phase 8.1 Active
            </span>
            <span className="text-slate-600 font-mono">v1.2</span>
          </div>
        </div>
      </aside>
    </>
  );
}
