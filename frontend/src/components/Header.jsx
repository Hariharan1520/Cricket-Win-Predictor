import React from 'react';
import { RefreshCw, Settings, Radio, Layers } from 'lucide-react';

export default function Header({
  isLive,
  isDemo,
  onRefresh,
  isRefreshing,
  activeMode,
  setActiveMode,
  pollCountdown,
  activeNav = 'live-matches',
  onNavClick,
}) {
  const navItems = [
    { id: 'live-matches', label: 'Live Matches' },
    { id: 'match-analysis', label: 'Match Analysis' },
    { id: 'swings', label: 'Swings' },
    { id: 'simulator', label: 'Simulator' },
  ];

  return (
    <header className="bg-white border-b border-[#E3EAF0] sticky top-0 z-40 px-4 sm:px-6 py-2.5 shadow-2xs">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        {/* Left: Brand with Cricket Ball Icon */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-[#EAF8F2] border border-[#0B9F72]/20 flex items-center justify-center text-[#0B9F72] shadow-2xs">
            <svg
              className="w-4 h-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10" />
              <path d="M12 2a15.3 15.3 0 0 0-4 10 15.3 15.3 0 0 0 4 10" />
            </svg>
          </div>
          <div>
            <div className="text-[13px] font-bold text-[#172B4D] tracking-tight leading-none">
              CRICKET WIN PREDICTOR
            </div>
            <div className="text-[9.5px] font-semibold text-[#8A98A8] tracking-widest uppercase mt-0.5 leading-none">
              T20 ANALYTICS
            </div>
          </div>
        </div>

        {/* Center: Navigation Links (Live Matches active by default) */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => {
            const isActive = (activeNav || 'live-matches') === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavClick && onNavClick(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? 'bg-[#EAF8F2] text-[#0B9F72] font-bold shadow-2xs'
                    : 'text-[#667085] hover:text-[#172B4D] hover:bg-[#F5F8FB]'
                }`}
              >
                {isActive && <span className="w-1.5 h-1.5 rounded-full bg-[#0B9F72]" />}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Controls: Compact Mode Switcher, Polling, Refresh, Settings */}
        <div className="flex items-center gap-2 sm:gap-2.5 flex-shrink-0">
          {/* Subtle Mode Switcher */}
          <div className="flex items-center bg-[#F5F8FB] p-0.5 rounded-lg border border-[#E3EAF0] text-xs">
            <button
              onClick={() => setActiveMode('live')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-bold transition ${
                activeMode === 'live'
                  ? 'bg-[#0B9F72] text-white shadow-2xs'
                  : 'text-[#667085] hover:text-[#172B4D]'
              }`}
            >
              <Radio className="w-3 h-3" />
              <span>LIVE</span>
            </button>
            <button
              onClick={() => setActiveMode('demo')}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-bold transition ${
                activeMode === 'demo'
                  ? 'bg-amber-600 text-white shadow-2xs'
                  : 'text-[#667085] hover:text-[#172B4D]'
              }`}
            >
              <Layers className="w-3 h-3" />
              <span>DEMO</span>
            </button>
          </div>

          {/* Polling Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#F5F8FB] border border-[#E3EAF0] text-[11px] font-mono text-[#667085]">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                activeMode === 'live' && isLive ? 'bg-[#0B9F72] animate-pulse' : 'bg-[#8A98A8]'
              }`}
            />
            <span>Poll {pollCountdown}s</span>
          </div>

          {/* Refresh Button */}
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="p-1.5 bg-white hover:bg-[#F5F8FB] border border-[#E3EAF0] text-[#667085] hover:text-[#172B4D] rounded-lg transition disabled:opacity-50"
            title="Refresh match data"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-[#0B9F72]' : ''}`} />
          </button>

          {/* Settings Icon */}
          <button
            className="p-1.5 bg-white hover:bg-[#F5F8FB] border border-[#E3EAF0] text-[#667085] hover:text-[#172B4D] rounded-lg transition"
            title="Settings"
          >
            <Settings className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
}
