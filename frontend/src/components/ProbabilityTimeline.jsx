import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

export default function ProbabilityTimeline({
  timeline,
  chasingTeam = 'India',
  defendingTeam = 'Australia',
}) {
  const rawData = timeline && timeline.length > 0
    ? timeline
    : [
        { over: 0, win_prob: 50.0, event: 'Start of Chase' },
      ];

  // Enrich data with defending team probability for dual-line visualization
  const data = rawData.map((pt) => ({
    ...pt,
    chasingProb: Number(pt.win_prob.toFixed(1)),
    defendingProb: Number((100.0 - pt.win_prob).toFixed(1)),
  }));

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      return (
        <div className="bg-white border border-[#E3EAF0] p-3 rounded-xl shadow-md text-xs font-sans">
          <div className="font-bold text-[#172B4D] mb-1">
            Over {point.over} {point.event ? `· ${point.event}` : ''}
          </div>
          <div className="flex items-center justify-between gap-5 py-0.5">
            <span className="flex items-center gap-1.5 font-semibold text-[#0B9F72]">
              <span className="w-2 h-2 rounded-full bg-[#0B9F72]" />
              {chasingTeam}:
            </span>
            <span className="font-mono font-black text-[#172B4D]">
              {point.chasingProb}%
            </span>
          </div>
          <div className="flex items-center justify-between gap-5 py-0.5">
            <span className="flex items-center gap-1.5 font-semibold text-[#EF5B67]">
              <span className="w-2 h-2 rounded-full bg-[#EF5B67]" />
              {defendingTeam}:
            </span>
            <span className="font-mono font-black text-[#172B4D]">
              {point.defendingProb}%
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white border border-[#E3EAF0] rounded-2xl p-5 sm:p-6 shadow-xs flex flex-col justify-between h-full">
      <div>
        {/* Header & Subtitle */}
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#172B4D] tracking-tight m-0">
              WIN PROBABILITY OVER TIME
            </h3>
            <p className="text-xs text-[#667085] m-0 mt-0.5">
              Model probability through the chase
            </p>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-3.5 text-xs font-semibold">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#0B9F72]" />
              <span className="text-[#172B4D]">{chasingTeam}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-[#EF5B67]" />
              <span className="text-[#172B4D]">{defendingTeam}</span>
            </div>
          </div>
        </div>

        {/* Chart Canvas */}
        <div className="h-64 sm:h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 15, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E3EAF0" vertical={false} />
              <XAxis
                dataKey="over"
                stroke="#8A98A8"
                fontSize={11}
                tickFormatter={(v) => `${v} ov`}
                domain={[0, 20]}
              />
              <YAxis
                stroke="#8A98A8"
                fontSize={11}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              {/* 50% Reference Par Line in #A8B4C2 */}
              <ReferenceLine
                y={50}
                stroke="#A8B4C2"
                strokeDasharray="4 4"
                label={{ value: '50% Par', fill: '#8A98A8', fontSize: 10, position: 'insideTopRight' }}
              />
              {/* Chasing Team Line (Green) */}
              <Line
                type="monotone"
                dataKey="chasingProb"
                name={chasingTeam}
                stroke="#0B9F72"
                strokeWidth={2.5}
                dot={{ r: 3.5, fill: '#0B9F72', stroke: '#FFFFFF', strokeWidth: 1.5 }}
                activeDot={{ r: 5, fill: '#0B9F72' }}
              />
              {/* Defending Team Line (Coral/Red) */}
              <Line
                type="monotone"
                dataKey="defendingProb"
                name={defendingTeam}
                stroke="#EF5B67"
                strokeWidth={2}
                strokeDasharray="2 2"
                dot={false}
                activeDot={{ r: 4, fill: '#EF5B67' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Axis Labels */}
      <div className="flex justify-between items-center text-[11px] text-[#8A98A8] mt-3 pt-2.5 border-t border-[#E3EAF0]">
        <span>Completed Overs (0 to 20)</span>
        <span>Chasing Win Probability (0% to 100%)</span>
      </div>
    </div>
  );
}
