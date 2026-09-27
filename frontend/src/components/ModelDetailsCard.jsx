import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';

export default function ModelDetailsCard() {
  const specs = [
    { label: 'MODEL', value: 'Logistic Regression' },
    { label: 'TEST ROC-AUC', value: '0.9109', highlight: true },
    { label: 'ARCHITECTURE', value: '120-Ball T20' },
    { label: 'LIVE FEED', value: 'Cricket Data API' },
    { label: 'FORMAT', value: 'T20 / T20I' },
    { label: 'BACKEND', value: 'Flask REST API' },
  ];

  return (
    <div className="bg-white border border-[#E3EAF0] rounded-2xl p-5 sm:p-6 shadow-xs h-full flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="w-4 h-4 text-[#0B9F72]" />
          <h3 className="text-sm font-bold text-[#172B4D] tracking-tight m-0">
            MODEL SPECIFICATIONS
          </h3>
        </div>
        <p className="text-xs text-[#667085] mb-4">
          Machine learning model calibrated on Cricsheet ball-by-ball historical chases.
        </p>

        {/* Specifications List */}
        <div className="space-y-2.5">
          {specs.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between py-1.5 border-b border-[#E3EAF0]/60 text-xs font-mono"
            >
              <span className="text-[#8A98A8] font-bold">{item.label}</span>
              <span
                className={`font-semibold ${
                  item.highlight ? 'text-[#0B9F72] font-black' : 'text-[#172B4D]'
                }`}
              >
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer verification note */}
      <div className="mt-4 pt-3 border-t border-[#E3EAF0] flex items-center gap-1.5 text-[11px] text-[#667085]">
        <ShieldCheck className="w-3.5 h-3.5 text-[#0B9F72] flex-shrink-0" />
        <span>Validated against historical Cricsheet second-innings dataset</span>
      </div>
    </div>
  );
}
