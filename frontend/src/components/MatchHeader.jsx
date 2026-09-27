import React from 'react';
import { MapPin, Target, Award } from 'lucide-react';

// Elegant cricket team emblem with authentic national colors
function TeamEmblem({ teamName, isChasing = false }) {
  const name = teamName || '';
  const clean = name.toLowerCase();

  let bgClass = 'bg-[#EAF3FA] text-[#2563EB] border-[#D0E2FF]';
  let shortCode = name.slice(0, 3).toUpperCase();
  let accentColor = '#2563EB';

  if (clean.includes('ind')) {
    bgClass = 'bg-[#00388A]/10 text-[#00388A] border-[#00388A]/25';
    shortCode = 'IND';
    accentColor = '#00388A';
  } else if (clean.includes('aus')) {
    bgClass = 'bg-[#004A26]/10 text-[#004A26] border-[#004A26]/25';
    shortCode = 'AUS';
    accentColor = '#004A26';
  } else if (clean.includes('eng')) {
    bgClass = 'bg-[#CC0000]/10 text-[#CC0000] border-[#CC0000]/25';
    shortCode = 'ENG';
    accentColor = '#CC0000';
  } else if (clean.includes('south') || clean.includes('sa')) {
    bgClass = 'bg-[#007749]/10 text-[#007749] border-[#007749]/25';
    shortCode = 'SA';
    accentColor = '#007749';
  } else if (clean.includes('pak')) {
    bgClass = 'bg-[#006629]/10 text-[#006629] border-[#006629]/25';
    shortCode = 'PAK';
    accentColor = '#006629';
  } else if (clean.includes('nz')) {
    bgClass = 'bg-slate-900/10 text-slate-900 border-slate-900/25';
    shortCode = 'NZ';
    accentColor = '#172B4D';
  }

  return (
    <div
      className={`w-14 h-14 sm:w-16 sm:h-16 rounded-2xl border flex flex-col items-center justify-center font-black text-sm sm:text-base tracking-wider shadow-xs flex-shrink-0 relative overflow-hidden ${bgClass}`}
    >
      {/* Subtle top indicator bar */}
      <div
        className="absolute top-0 left-0 right-0 h-1"
        style={{ backgroundColor: accentColor }}
      />
      <span>{shortCode}</span>
    </div>
  );
}

export default function MatchHeader({ match }) {
  if (!match) return null;

  const {
    match_name,
    chasing_team = 'India',
    defending_team = 'Australia',
    current_score = 142,
    wickets_lost = 4,
    overs_display = '16.0 ov',
    target_score = 190,
    runs_remaining = 48,
    balls_remaining = 24,
    current_run_rate = 8.88,
    required_run_rate = 12.0,
    status = '',
    venue = 'Kensington Oval, Bridgetown, Barbados',
    is_terminal = false,
    terminal_state = '',
    format = 'T20I',
  } = match;

  const targetRounded = Math.round(target_score);
  const runsRemainingRounded = Math.round(runs_remaining);
  const defendingScore = Math.max(0, targetRounded - 1);

  // Format overs display: clean up to "16.0 OVERS"
  const oversClean = overs_display.split('(')[0].replace('ov', '').trim();

  return (
    <div
      id="match-analysis"
      className="bg-white border border-[#E3EAF0] rounded-2xl p-6 sm:p-7 mb-5 shadow-xs relative overflow-hidden"
      style={{
        background:
          'radial-gradient(ellipse at 15% 20%, rgba(234, 248, 242, 0.45) 0%, transparent 50%), radial-gradient(ellipse at 85% 20%, rgba(238, 245, 252, 0.5) 0%, transparent 50%), #FFFFFF',
      }}
    >
      {/* Subtle Stadium/Cricket Ground Pitch Silhouette Texture */}
      <svg
        className="absolute right-0 top-0 bottom-0 h-full w-auto opacity-[0.035] pointer-events-none text-[#172B4D]"
        viewBox="0 0 400 200"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <ellipse cx="200" cy="100" rx="190" ry="90" />
        <ellipse cx="200" cy="100" rx="140" ry="65" strokeDasharray="6 6" />
        <rect x="175" y="70" width="50" height="60" rx="2" />
        <line x1="175" y1="80" x2="225" y2="80" />
        <line x1="175" y1="120" x2="225" y2="120" />
      </svg>

      {/* Top Meta Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E3EAF0] pb-3 mb-6 text-xs text-[#667085]">
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded text-[11px] font-bold uppercase bg-[#EAF8F2] text-[#0B9F72]">
            {format} · SUPER 8
          </span>
          <span className="font-semibold text-[#172B4D]">{match_name}</span>
        </div>
        {venue && (
          <div className="flex items-center gap-1.5 text-[11px] text-[#667085]">
            <MapPin className="w-3.5 h-3.5 text-[#8A98A8]" />
            <span className="truncate max-w-[340px]">{venue}</span>
          </div>
        )}
      </div>

      {/* Main Scoreboard: Left (Chasing) - Center (Anchor Target) - Right (Defending) */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center relative z-10">
        {/* Left: Chasing Team */}
        <div className="md:col-span-5 flex items-center gap-4 sm:gap-5">
          <TeamEmblem teamName={chasing_team} isChasing={true} />
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight">
                {chasing_team.toUpperCase()}
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-[#EAF8F2] text-[#0B9F72] border border-[#0B9F72]/20">
                CHASING
              </span>
            </div>
            <div className="flex items-baseline gap-3">
              <span className="text-4xl sm:text-5xl font-mono font-black text-[#172B4D] tracking-tight tabular-nums">
                {Math.round(current_score)} / {wickets_lost}
              </span>
              <span className="text-sm font-mono font-bold text-[#667085]">
                {oversClean} OVERS
              </span>
            </div>
          </div>
        </div>

        {/* Center: Target Centerpiece (Anchor with #EEF5FC soft background) */}
        <div className="md:col-span-2 flex flex-col items-center justify-center p-3.5 rounded-2xl bg-[#EEF5FC] border border-[#D0E2FF] text-center shadow-xs">
          <div className="text-[10px] font-bold uppercase tracking-wider text-[#667085] mb-0.5">
            TARGET
          </div>
          <div className="text-3xl font-mono font-black text-[#172B4D] tabular-nums leading-none">
            {targetRounded}
          </div>
          <div className="text-[11px] font-bold text-[#0B9F72] mt-2 uppercase leading-tight">
            {chasing_team.toUpperCase()} NEED
          </div>
          <div className="text-xs font-black text-[#172B4D] mt-0.5 leading-tight">
            {runsRemainingRounded} RUNS FROM {balls_remaining} BALLS
          </div>
          <div className="flex items-center gap-2 text-[10.5px] font-mono font-semibold text-[#667085] mt-2 pt-1.5 border-t border-[#D0E2FF] w-full justify-center">
            <span>CRR {current_run_rate.toFixed(2)}</span>
            <span>·</span>
            <span>RRR {required_run_rate.toFixed(2)}</span>
          </div>
        </div>

        {/* Right: Defending Team */}
        <div className="md:col-span-5 flex md:flex-row-reverse items-center gap-4 sm:gap-5 md:text-right">
          <TeamEmblem teamName={defending_team} isChasing={false} />
          <div>
            <div className="flex items-center md:justify-end gap-2 mb-1">
              <span className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight">
                {defending_team.toUpperCase()}
              </span>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-[#FFF0F2] text-[#EF5B67] border border-[#EF5B67]/20">
                DEFENDING
              </span>
            </div>
            <div className="flex items-baseline md:justify-end gap-3">
              <span className="text-4xl sm:text-5xl font-mono font-black text-[#172B4D] tracking-tight tabular-nums">
                {defendingScore} / 6
              </span>
              <span className="text-sm font-mono font-bold text-[#667085]">
                20.0 OVERS
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Terminal State Alert */}
      {is_terminal && (
        <div className="mt-5 pt-3 border-t border-[#E3EAF0] flex items-center gap-2 text-xs font-bold text-amber-800 bg-amber-50 p-2.5 rounded-xl border border-amber-200">
          <Award className="w-4 h-4 text-amber-700 flex-shrink-0" />
          <span>MATCH CONCLUDED: {terminal_state}</span>
        </div>
      )}
    </div>
  );
}
