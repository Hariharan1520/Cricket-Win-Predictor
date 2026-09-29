import React, { useState, useEffect, useRef } from 'react';
import { RefreshCw, Settings, Radio, Layers, X, Clock } from 'lucide-react';

// ─── Settings Modal ───────────────────────────────────────────────────────────
function SettingsModal({ isOpen, onClose, pollInterval, onPollIntervalChange }) {
  const overlayRef = useRef(null);

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  // Close on outside click
  const handleOverlayClick = (e) => {
    if (e.target === overlayRef.current) onClose();
  };

  if (!isOpen) return null;

  const intervalOptions = [
    { value: 15, label: '15 seconds' },
    { value: 30, label: '30 seconds' },
    { value: 60, label: '60 seconds' },
    { value: 120, label: '2 minutes' },
  ];

  return (
    <div
      ref={overlayRef}
      onClick={handleOverlayClick}
      className="fixed inset-0 z-50 flex items-start justify-end"
      style={{ background: 'rgba(23,43,77,0.15)', backdropFilter: 'blur(2px)' }}
    >
      {/* Panel — top-right, beneath the header */}
      <div
        className="mt-14 mr-4 w-72 bg-white rounded-2xl border border-[#E3EAF0] shadow-xl overflow-hidden"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
      >
        {/* Header bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#E3EAF0] bg-[#F5F8FB]">
          <div className="flex items-center gap-2">
            <Settings className="w-3.5 h-3.5 text-[#0B9F72]" />
            <span className="text-sm font-bold text-[#172B4D] tracking-tight">Settings</span>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-[#8A98A8] hover:text-[#172B4D] hover:bg-[#E3EAF0] transition"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Body */}
        <div className="px-4 py-4 space-y-5">
          {/* Polling Interval */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Clock className="w-3.5 h-3.5 text-[#667085]" />
              <span className="text-xs font-bold text-[#172B4D] uppercase tracking-wide">
                Live Polling Interval
              </span>
            </div>
            <p className="text-[11px] text-[#8A98A8] mb-3 leading-relaxed">
              How often the dashboard refreshes live match data. Applies only in Live mode.
            </p>
            <div className="grid grid-cols-2 gap-2">
              {intervalOptions.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => onPollIntervalChange(value)}
                  className={`px-3 py-2 rounded-lg border text-xs font-semibold transition ${
                    pollInterval === value
                      ? 'bg-[#EAF8F2] border-[#0B9F72] text-[#0B9F72] shadow-sm'
                      : 'bg-white border-[#E3EAF0] text-[#667085] hover:bg-[#F5F8FB] hover:border-[#D0D9E2]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-[#E3EAF0]" />

          {/* Info note */}
          <div className="text-[11px] text-[#8A98A8] leading-relaxed">
            Recent mode is database-only — polling interval does not apply when browsing archived matches.
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Header ───────────────────────────────────────────────────────────────────
export default function Header({
  isLive,
  isDemo,
  onRefresh,
  isRefreshing,
  activeMode,
  setActiveMode,
  pollInterval,
  onPollIntervalChange,
  pollCountdown,
  activeNav = 'live-matches',
  onNavClick,
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);

  const navItems = [
    { id: 'live-matches', label: 'Live Matches' },
    { id: 'match-analysis', label: 'Match Analysis' },
    { id: 'swings', label: 'Swings' },
    { id: 'simulator', label: 'Simulator' },
  ];

  return (
    <>
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

          {/* Center: Navigation Links */}
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

          {/* Right Controls */}
          <div className="flex items-center gap-2 sm:gap-2.5 flex-shrink-0">
            {/* Mode Switcher */}
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
                onClick={() => setActiveMode('recent')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-bold transition ${
                  activeMode === 'recent' || activeMode === 'demo'
                    ? 'bg-[#186ADE] text-white shadow-2xs'
                    : 'text-[#667085] hover:text-[#172B4D]'
                }`}
              >
                <Layers className="w-3 h-3" />
                <span>RECENT</span>
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

            {/* Settings Button — now wired up */}
            <button
              onClick={() => setSettingsOpen(true)}
              className={`p-1.5 border rounded-lg transition ${
                settingsOpen
                  ? 'bg-[#EAF8F2] border-[#0B9F72] text-[#0B9F72]'
                  : 'bg-white hover:bg-[#F5F8FB] border-[#E3EAF0] text-[#667085] hover:text-[#172B4D]'
              }`}
              title="Settings"
              aria-expanded={settingsOpen}
            >
              <Settings className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Modal — rendered outside header flow so it doesn't clip */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        pollInterval={pollInterval}
        onPollIntervalChange={(v) => {
          onPollIntervalChange(v);
          setSettingsOpen(false);
        }}
      />
    </>
  );
}
