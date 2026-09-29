import React from 'react';

// ─── MatchSelector ────────────────────────────────────────────────────────────
export default function MatchSelector({
  matches,
  selectedMatchId,
  onSelectMatch,
  title = 'Live Matches',
  isDemo = false,
}) {
  if (!matches || matches.length === 0) return null;

  const getMatchSituation = (m) => {
    // If demo match, show exact realistic chase situations
    if (m.match_id === 'demo-match-1') {
      return {
        teams: 'AUSTRALIA vs INDIA',
        liveMeta: 'T20 · DEMO · 16.0 OV',
        situation: 'SIMULATED: India need 48 from 24',
      };
    }
    if (m.match_id === 'demo-match-2') {
      return {
        teams: 'ENGLAND vs SOUTH AFRICA',
        liveMeta: 'T20 · DEMO · 17.0 OV',
        situation: 'SIMULATED: South Africa need 28 from 18',
      };
    }

    const teams = m.teams || (m.name ? m.name.split(' vs ') : ['Team A', 'Team B']);
    const teamA = (teams[0] || 'Team A').toUpperCase();
    const teamB = (teams[1] || 'Team B').toUpperCase();
    const format = m.format || 'T20';

    if (m.is_recent) {
      return {
        teams: `${teamA} vs ${teamB}`,
        liveMeta: `${format} · ${m.analysis_available ? 'REPLAY READY' : 'ARCHIVED'}`,
        situation: m.status || (m.winner ? `${m.winner} won` : 'Completed'),
      };
    }

    return {
      teams: `${teamA} vs ${teamB}`,
      liveMeta: `${format} · LIVE`,
      situation: m.status || 'Chase in progress',
    };
  };

  return (
    <div id="live-matches" className="bg-white border-b border-[#E3EAF0] py-2.5 px-4 sm:px-6">
      <div className="max-w-7xl mx-auto">
        {/* Horizontal Match Strip */}
        <div className="flex items-center gap-2.5 overflow-x-auto pb-1 scrollbar-none">
          {matches.map((m) => {
            const isSelected = m.match_id === selectedMatchId;
            const info = getMatchSituation(m);

            return (
              <button
                key={m.match_id}
                onClick={() => onSelectMatch(m.match_id)}
                className={`flex-shrink-0 text-left px-3.5 py-2 rounded-xl border transition-all ${
                  isSelected
                    ? 'bg-[#EAF8F2] border-[#0B9F72] shadow-2xs'
                    : 'bg-white border-[#E3EAF0] hover:bg-[#F5F8FB] hover:border-[#D0D9E2]'
                }`}
              >
                {/* Teams Line */}
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      isSelected ? 'bg-[#0B9F72]' : 'bg-[#8A98A8]'
                    }`}
                  />
                  <span className="text-xs font-bold text-[#172B4D] tracking-tight whitespace-nowrap">
                    {info.teams}
                  </span>
                </div>

                {/* Live State & Overs */}
                <div className="text-[10.5px] font-mono font-semibold text-[#8A98A8] mt-0.5 pl-3.5">
                  {info.liveMeta}
                </div>

                {/* Chase Equation / Result */}
                <div className="text-[11px] font-semibold text-[#0B9F72] mt-0.5 pl-3.5 whitespace-nowrap">
                  {info.situation}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
