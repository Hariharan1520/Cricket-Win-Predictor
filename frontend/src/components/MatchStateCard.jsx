import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Layers, CheckCircle2 } from 'lucide-react';

export default function MatchStateCard({ match }) {
  const [showFeatures, setShowFeatures] = useState(false);

  if (!match) return null;

  const {
    current_score = 142,
    wickets_lost = 4,
    target_score = 190,
    runs_remaining = 48,
    balls_remaining = 24,
    overs_completed = 16.0,
    current_run_rate = 8.88,
    required_run_rate = 12.0,
  } = match;

  const stats = [
    {
      label: 'CURRENT SCORE',
      value: `${Math.round(current_score)} / ${wickets_lost}`,
      sub: `${overs_completed.toFixed(1)} ov`,
    },
    {
      label: 'TARGET',
      value: `${Math.round(target_score)}`,
      sub: 'Runs to win',
    },
    {
      label: 'RUNS NEEDED',
      value: `${Math.round(runs_remaining)}`,
      sub: `From ${balls_remaining} balls`,
      highlight: true,
    },
    {
      label: 'BALLS REMAINING',
      value: `${balls_remaining}`,
      sub: `${(balls_remaining / 6).toFixed(1)} overs left`,
    },
    {
      label: 'CURRENT RR',
      value: current_run_rate.toFixed(2),
      sub: 'Runs / over',
    },
    {
      label: 'REQUIRED RR',
      value: required_run_rate.toFixed(2),
      sub: required_run_rate > 10.0 ? 'High pressure' : 'Manageable',
      color: required_run_rate > 10.0 ? 'text-[#EF5B67]' : 'text-[#0B9F72]',
    },
  ];

  // The 8 exact raw numerical features ingested by Phase 3 ML pipeline
  const modelFeatures = [
    { name: 'target_score', value: target_score.toFixed(1), desc: 'Target runs to win chase' },
    { name: 'current_score', value: current_score.toFixed(1), desc: 'Cumulative chasing runs' },
    { name: 'wickets_lost', value: wickets_lost.toString(), desc: 'Total wickets fallen (0-10)' },
    { name: 'runs_remaining', value: runs_remaining.toFixed(1), desc: 'Target minus current score' },
    { name: 'balls_remaining', value: balls_remaining.toString(), desc: '120 minus legal balls bowled' },
    { name: 'overs_completed', value: overs_completed.toFixed(2), desc: 'Fractional completed overs' },
    { name: 'current_run_rate', value: current_run_rate.toFixed(2), desc: 'Score / overs_completed' },
    { name: 'required_run_rate', value: required_run_rate.toFixed(2), desc: 'Runs remaining / overs remaining' },
  ];

  return (
    <div className="bg-white border border-[#E3EAF0] rounded-2xl p-4 sm:p-5 mb-5 shadow-xs">
      {/* Horizontal Stats Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4 items-center">
        {stats.map((item, idx) => (
          <div
            key={idx}
            className={`flex flex-col justify-center px-2 py-1 ${
              idx !== stats.length - 1 ? 'lg:border-r border-[#E3EAF0]' : ''
            }`}
          >
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8A98A8] mb-1">
              {item.label}
            </div>
            <div
              className={`text-2xl sm:text-3xl font-mono font-black ${
                item.color || (item.highlight ? 'text-[#0B9F72]' : 'text-[#172B4D]')
              } tracking-tight tabular-nums`}
            >
              {item.value}
            </div>
            <div className="text-[11px] font-semibold text-[#667085] mt-0.5">
              {item.sub}
            </div>
          </div>
        ))}
      </div>

      {/* Expandable Section at Right / Bottom */}
      <div className="mt-3.5 pt-3 border-t border-[#E3EAF0] flex flex-col sm:flex-row items-center justify-between gap-2">
        <span className="text-xs text-[#667085]">
          Real-time match situation metrics calibrated for 120-ball chase model.
        </span>

        {/* 8 MODEL FEATURES USED Dropdown Control */}
        <button
          onClick={() => setShowFeatures(!showFeatures)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#F5F8FB] hover:bg-[#EEF5FC] border border-[#E3EAF0] text-xs font-semibold text-[#172B4D] transition shadow-2xs"
        >
          <Layers className="w-3.5 h-3.5 text-[#0B9F72]" />
          <span>8 MODEL FEATURES USED</span>
          {showFeatures ? (
            <ChevronUp className="w-3.5 h-3.5 text-[#8A98A8]" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-[#8A98A8]" />
          )}
        </button>
      </div>

      {/* Expanded Features Drawer */}
      {showFeatures && (
        <div className="mt-3 p-3.5 bg-[#F5F8FB] border border-[#E3EAF0] rounded-xl animate-in fade-in duration-200">
          <div className="text-xs font-semibold text-[#172B4D] mb-2.5 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#0B9F72]" />
            <span>Exact raw input vector passed to StandardScaler + LogisticRegression pipeline:</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {modelFeatures.map((feat, idx) => (
              <div
                key={idx}
                className="bg-white border border-[#E3EAF0] rounded-lg p-2.5 shadow-2xs"
              >
                <div className="text-[11px] font-mono font-bold text-[#667085] truncate">
                  {feat.name}
                </div>
                <div className="text-lg font-mono font-black text-[#172B4D] mt-0.5 tabular-nums">
                  {feat.value}
                </div>
                <div className="text-[10px] text-[#8A98A8] truncate mt-0.5">
                  {feat.desc}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
