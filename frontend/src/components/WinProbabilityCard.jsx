import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function WinProbabilityCard({ match, isRecent = false, isDemo = false }) {
  if (!match) return null;

  const isHistorical = isRecent || match.is_recent;
  const chasingTeam = match.chasing_team || 'India';
  const defendingTeam = match.defending_team || 'Australia';
  const chasingProb = match.win_probability_pct ?? 40.6;
  const defendingProb = match.loss_probability_pct ?? (100.0 - chasingProb);
  const probChange = match.win_probability_change;

  const hasSwing = probChange !== undefined && probChange !== null;
  const swingVal = hasSwing ? probChange * 100 : 0;
  const isPos = swingVal > 0;
  const isNeg = swingVal < 0;

  return (
    <div className="bg-white border border-[#E3EAF0] rounded-2xl p-5 sm:p-6 mb-5 shadow-xs">
      {/* Title */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-[#8A98A8] m-0">
          {isDemo ? 'DEMO WIN PROBABILITY' : isHistorical ? 'HISTORICAL WIN PROBABILITY' : 'WIN PROBABILITY'}
        </h3>
        <span className="text-[11px] font-mono text-[#8A98A8]">
          CALIBRATED 120-BALL INFERENCE
        </span>
      </div>

      {/* Main Team Probability Numbers */}
      <div className="grid grid-cols-2 gap-4 items-end mb-2.5">
        <div>
          <div className="text-sm font-bold text-[#172B4D]">
            {chasingTeam}
          </div>
          <div className="text-4xl sm:text-5xl font-mono font-black text-[#0B9F72] tracking-tight tabular-nums">
            {chasingProb.toFixed(1)}%
          </div>
        </div>

        <div className="text-right">
          <div className="text-sm font-bold text-[#172B4D]">
            {defendingTeam}
          </div>
          <div className="text-4xl sm:text-5xl font-mono font-black text-[#EF5B67] tracking-tight tabular-nums">
            {defendingProb.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Proportional Dual Probability Bar */}
      <div className="w-full h-3.5 bg-[#F5F8FB] rounded-full overflow-hidden flex border border-[#E3EAF0] p-0.5 mb-3.5">
        <div
          className="h-full bg-[#0B9F72] rounded-l-full transition-all duration-500 ease-out"
          style={{ width: `${Math.max(2, Math.min(98, chasingProb))}%` }}
        />
        <div
          className="h-full bg-[#EF5B67] rounded-r-full transition-all duration-500 ease-out"
          style={{ width: `${Math.max(2, Math.min(98, defendingProb))}%` }}
        />
      </div>

      {/* Integrated Latest Swing Indicator & Context Footer */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 pt-2.5 border-t border-[#E3EAF0]/80">
        <div className="flex items-center gap-2">
          {hasSwing && (
            <div
              className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-mono font-bold ${
                isPos
                  ? 'bg-[#EAF8F2] text-[#0B9F72] border border-[#0B9F72]/30'
                  : isNeg
                  ? 'bg-[#FFF0F2] text-[#EF5B67] border border-[#EF5B67]/30'
                  : 'bg-[#F5F8FB] text-[#667085] border border-[#E3EAF0]'
              }`}
            >
              {isPos ? (
                <TrendingUp className="w-3.5 h-3.5" />
              ) : isNeg ? (
                <TrendingDown className="w-3.5 h-3.5" />
              ) : (
                <Minus className="w-3.5 h-3.5" />
              )}
              <span>
                LATEST SWING {isPos ? '↑ ' : isNeg ? '↓ ' : ''}
                {Math.abs(swingVal).toFixed(1)}%
              </span>
            </div>
          )}
          <span className="text-xs text-[#667085]">
            Probability change from latest recorded event.
          </span>
        </div>

        <span className="text-[11px] font-mono text-[#8A98A8]">
          Target {Math.round(match.target_score || 0)} · Second Innings
        </span>
      </div>
    </div>
  );
}
