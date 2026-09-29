import React, { useState } from 'react';
import { ArrowRight, Check, HelpCircle } from 'lucide-react';

export default function NextOverSimulator({
  scenarios,
  currentWinProbPct,
  isRecent = false,
}) {
  const [selectedIdx, setSelectedIdx] = useState(5); // Default to mid scenario (e.g. 6 or 8 runs)
  const [showAllScenarios, setShowAllScenarios] = useState(false);

  if (!scenarios || scenarios.length === 0) return null;

  const currentScenario = scenarios[selectedIdx] || scenarios[0];
  const curProb = currentWinProbPct ?? (currentScenario?.current_win_probability ? currentScenario.current_win_probability * 100 : 40.6);
  const simProb = currentScenario ? (currentScenario.simulated_win_probability * 100).toFixed(1) : '40.6';
  const probChange = currentScenario ? (currentScenario.probability_change * 100).toFixed(1) : '0.0';
  const isPos = parseFloat(probChange) >= 0;

  // Decide whether to show 6 primary scenarios or all 11
  const displayedScenarios = showAllScenarios ? scenarios : scenarios.slice(0, 6);

  return (
    <div id="simulator" className="bg-white border border-[#E3EAF0] rounded-2xl p-5 sm:p-6 shadow-xs h-full flex flex-col justify-between">
      <div>
        {/* Header & Subtitle */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#172B4D] tracking-tight m-0">
              WHAT-IF NEXT-OVER SIMULATION
            </h3>
            <p className="text-xs text-[#667085] m-0 mt-0.5">
              {isRecent
                ? 'Hypothetical next-over simulation from stored match state · Not an actual historical event.'
                : 'Explore how different next-over outcomes could change predicted win probability.'}
            </p>
          </div>

          {/* Current Probability Pill */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#F5F8FB] border border-[#E3EAF0] self-start sm:self-auto">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[#8A98A8]">
              CURRENT PROBABILITY:
            </span>
            <span className="text-sm font-mono font-black text-[#172B4D]">
              {curProb.toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Selected Scenario Impact Banner */}
        <div className="bg-[#EEF5FC] border border-[#D0E2FF] rounded-xl p-3.5 mb-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-center sm:text-left">
            <div>
              <div className="text-[11px] font-bold text-[#0B9F72] uppercase tracking-wider">
                SELECTED SCENARIO: {currentScenario?.scenario_name}
              </div>
              <div className="text-xs text-[#667085] mt-0.5">
                Outcome after 6 legal deliveries: +{currentScenario?.scenario_runs} runs
                {currentScenario?.scenario_wickets > 0 ? ` · ${currentScenario?.scenario_wickets} wicket lost` : ' · 0 wickets'}
              </div>
            </div>

            <div className="flex items-center justify-center sm:justify-end gap-2.5">
              <span className="text-xs font-mono text-[#8A98A8]">{curProb.toFixed(1)}%</span>
              <ArrowRight className="w-3.5 h-3.5 text-[#8A98A8]" />
              <div className="flex items-baseline gap-1.5">
                <span className="text-xl font-mono font-black text-[#172B4D]">
                  {simProb}%
                </span>
                <span
                  className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded ${
                    isPos
                      ? 'bg-[#EAF8F2] text-[#0B9F72]'
                      : 'bg-[#FFF0F2] text-[#EF5B67]'
                  }`}
                >
                  {isPos ? '↑ ' : '↓ '}
                  {Math.abs(parseFloat(probChange)).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Scenario Cards Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 gap-2.5 mb-3">
          {displayedScenarios.map((sc, idx) => {
            const isSelected = idx === selectedIdx;
            const delta = (sc.probability_change * 100).toFixed(1);
            const isDeltaPos = parseFloat(delta) >= 0;
            const projProb = (sc.simulated_win_probability * 100).toFixed(1);

            return (
              <button
                key={idx}
                onClick={() => setSelectedIdx(idx)}
                className={`p-3 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-[#EAF8F2] border-[#0B9F72] shadow-xs'
                    : 'bg-white border-[#E3EAF0] hover:bg-[#F5F8FB] hover:border-[#D0D9E2]'
                }`}
              >
                <div className="flex items-center justify-between text-xs font-bold text-[#172B4D]">
                  <span>
                    {sc.scenario_wickets > 0
                      ? `${sc.scenario_runs} RUNS, ${sc.scenario_wickets}W`
                      : `${sc.scenario_runs} RUNS`}
                  </span>
                  {isSelected && <Check className="w-3.5 h-3.5 text-[#0B9F72]" />}
                </div>

                <div className="flex items-baseline justify-between mt-1.5">
                  <span className="text-base font-mono font-black text-[#172B4D] tabular-nums">
                    {projProb}%
                  </span>
                  <span
                    className={`text-xs font-mono font-bold ${
                      isDeltaPos ? 'text-[#0B9F72]' : 'text-[#EF5B67]'
                    }`}
                  >
                    {isDeltaPos ? '↑ ' : '↓ '}
                    {Math.abs(parseFloat(delta)).toFixed(1)}%
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        {/* View All 11 Scenarios Toggle */}
        <div className="flex justify-end">
          <button
            onClick={() => setShowAllScenarios(!showAllScenarios)}
            className="text-xs font-semibold text-[#0B9F72] hover:text-[#168A5B] transition py-1"
          >
            {showAllScenarios ? 'Show 6 primary scenarios' : `View all ${scenarios.length} scenarios`}
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-4 pt-3 border-t border-[#E3EAF0] flex items-center justify-between text-[11px] text-[#8A98A8]">
        <span className="flex items-center gap-1.5">
          <HelpCircle className="w-3.5 h-3.5 text-[#8A98A8]" />
          Scenario sensitivity only — not an exact next-over prediction.
        </span>
        <span className="hidden sm:inline font-mono">Phase 6 Engine</span>
      </div>
    </div>
  );
}
