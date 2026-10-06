import React from 'react';
import { ArrowUpRight, ArrowDownRight, Minus, AlertCircle } from 'lucide-react';

// Cricket-specific event icon badge
function EventBadge({ label }) {
  const clean = (label || '').toLowerCase();

  if (clean.includes('wicket') || clean.includes('bowled') || clean.includes('caught')) {
    return (
      <span className="w-7 h-7 rounded-lg bg-[#FFF0F2] text-[#EF5B67] border border-[#EF5B67]/25 flex items-center justify-center font-black text-xs flex-shrink-0">
        W
      </span>
    );
  }
  if (clean.includes('six') || clean.includes('maximum')) {
    return (
      <span className="w-7 h-7 rounded-lg bg-[#EAF8F2] text-[#0B9F72] border border-[#0B9F72]/25 flex items-center justify-center font-black text-xs flex-shrink-0">
        6
      </span>
    );
  }
  if (clean.includes('four') || clean.includes('boundary')) {
    return (
      <span className="w-7 h-7 rounded-lg bg-[#EAF8F2] text-[#0B9F72] border border-[#0B9F72]/25 flex items-center justify-center font-black text-xs flex-shrink-0">
        4
      </span>
    );
  }
  if (clean.includes('dot')) {
    return (
      <span className="w-7 h-7 rounded-lg bg-[#F5F8FB] text-[#8A98A8] border border-[#E3EAF0] flex items-center justify-center font-black text-xs flex-shrink-0">
        •
      </span>
    );
  }
  return (
    <span className="w-7 h-7 rounded-lg bg-[#EEF5FC] text-[#2563EB] border border-[#D0E2FF] flex items-center justify-center font-black text-xs flex-shrink-0">
      1
    </span>
  );
}

export default function ProbabilitySwings({
  recentSwings,
  isRecent = false,
  chasingTeam = 'Chasing Team',
  defendingTeam = 'Defending Team',
}) {
  const defaultSwings = [
    { delivery: '14.3', event: 'FOUR', swing: '+4.8 pp', type: 'positive', benefiting_team: chasingTeam },
    { delivery: '13.5', event: 'WICKET', swing: '-7.4 pp', type: 'negative', benefiting_team: defendingTeam },
    { delivery: '12.6', event: 'SIX', swing: '+8.2 pp', type: 'positive', benefiting_team: chasingTeam },
    { delivery: '11.2', event: 'DOT BALL', swing: '-1.8 pp', type: 'negative', benefiting_team: defendingTeam },
  ];

  const swings = recentSwings && recentSwings.length > 0
    ? recentSwings
    : isRecent
      ? []
      : defaultSwings;

  return (
    <div id="swings" className="bg-white border border-[#E3EAF0] rounded-2xl p-5 sm:p-6 shadow-xs flex flex-col justify-between h-full">
      <div>
        {/* Header & Subtitle */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#172B4D] tracking-tight m-0">
              {isRecent ? 'KEY TURNING POINTS' : 'KEY PROBABILITY SWINGS'}
            </h3>
            <p className="text-xs text-[#667085] m-0 mt-0.5">
              {isRecent ? 'Largest match-defining probability shifts' : 'Largest model probability changes'}
            </p>
          </div>
          <span className="text-[11px] font-mono text-[#8A98A8]">
            OVER · DELTA (pp)
          </span>
        </div>

        {/* Rows */}
        <div className="space-y-2.5">
          {isRecent && swings.length === 0 && (
            <p className="rounded-lg border border-[#E3EAF0] bg-[#F5F8FB] px-3 py-4 text-xs text-[#667085]">
              No historical swing events are available.
            </p>
          )}
          {swings.map((s, idx) => {
            const isPos = s.type === 'positive' || (typeof s.swing === 'string' && s.swing.startsWith('+'));
            const isNeg = s.type === 'negative' || (typeof s.swing === 'string' && s.swing.startsWith('-'));

            // Ensure pp unit formatting
            let displaySwing = s.swing;
            if (typeof displaySwing === 'string') {
              if (displaySwing.endsWith('%')) {
                displaySwing = displaySwing.replace('%', ' pp');
              } else if (!displaySwing.includes('pp')) {
                displaySwing = `${displaySwing} pp`;
              }
            }

            const teamBenefited = s.benefiting_team || (isPos ? chasingTeam : isNeg ? defendingTeam : null);

            return (
              <div
                key={idx}
                className="flex items-center justify-between p-2.5 rounded-xl bg-[#F5F8FB] hover:bg-[#EEF5FC] transition border border-[#E3EAF0]/60"
              >
                {/* Event info */}
                <div className="flex items-center gap-3">
                  <EventBadge label={s.event} />
                  <div>
                    <div className="text-xs font-bold text-[#172B4D] leading-tight">
                      {s.event.toUpperCase()}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[11px] text-[#8A98A8] font-mono">
                        Over {s.delivery}
                      </span>
                      {teamBenefited && (
                        <span className={`text-[10px] font-semibold px-1.5 py-0.2 rounded border ${
                          isPos
                            ? 'bg-[#EAF8F2] text-[#0B9F72] border-[#0B9F72]/20'
                            : isNeg
                            ? 'bg-[#FFF0F2] text-[#EF5B67] border-[#EF5B67]/20'
                            : 'bg-white text-[#667085] border-[#E3EAF0]'
                        }`}>
                          {teamBenefited} benefit
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Delta Badge */}
                <div
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono font-bold ${
                    isPos
                      ? 'bg-[#EAF8F2] text-[#0B9F72]'
                      : isNeg
                      ? 'bg-[#FFF0F2] text-[#EF5B67]'
                      : 'bg-white text-[#667085] border border-[#E3EAF0]'
                  }`}
                >
                  <span>{displaySwing}</span>
                  {isPos ? (
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  ) : isNeg ? (
                    <ArrowDownRight className="w-3.5 h-3.5" />
                  ) : (
                    <Minus className="w-3.5 h-3.5" />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Explanatory Note */}
      <div className="mt-4 pt-3 border-t border-[#E3EAF0] flex items-start gap-1.5 text-[11px] text-[#8A98A8] leading-relaxed">
        <AlertCircle className="w-3.5 h-3.5 text-[#8A98A8] flex-shrink-0 mt-0.5" />
        <p className="m-0">
          Probability swings represent model probability changes between events; they are not a direct measure of psychological or team momentum.
        </p>
      </div>
    </div>
  );
}
