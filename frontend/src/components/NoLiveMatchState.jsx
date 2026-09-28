import React from 'react';
import { Radio, RefreshCw, Layers, ShieldCheck, Filter } from 'lucide-react';

export default function NoLiveMatchState({ onSwitchToRecent, onSwitchToDemo, onRefresh, isRefreshing }) {
  return (
    <div className="bg-white border border-[#E3EAF0] rounded-2xl p-8 sm:p-12 text-center max-w-2xl mx-auto my-8 shadow-xs">
      {/* Icon */}
      <div className="w-14 h-14 rounded-full bg-[#EAF8F2] border border-[#0B9F72]/30 flex items-center justify-center mx-auto mb-4 text-[#0B9F72]">
        <Radio className="w-7 h-7" />
      </div>

      {/* Title */}
      <h2 className="text-xl sm:text-2xl font-bold text-[#172B4D] tracking-tight mb-2">
        NO LIVE T20 MATCH AVAILABLE
      </h2>

      {/* Description */}
      <p className="text-xs sm:text-sm text-[#667085] max-w-lg mx-auto leading-relaxed mb-5">
        There is currently no active T20 or T20I match in progress on the live Cricket Data API feed.
        The dashboard is calibrated and standing by to ingest live deliveries for the next standard T20 chase.
      </p>

      {/* Filter note */}
      <div className="mb-6 inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-[#F5F8FB] border border-[#E3EAF0] text-xs text-[#667085]">
        <Filter className="w-3.5 h-3.5 text-[#0B9F72]" />
        <span>Strict T20-only filtering active: Test and ODI matches are ignored.</span>
      </div>

      {/* Actions */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="w-full sm:w-auto px-4 py-2 rounded-xl bg-white hover:bg-[#F5F8FB] border border-[#E3EAF0] text-[#172B4D] text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-[#0B9F72]' : 'text-[#667085]'}`} />
          CHECK LIVE FEED AGAIN
        </button>

        <button
          onClick={onSwitchToRecent || onSwitchToDemo}
          className="w-full sm:w-auto px-4 py-2 rounded-xl bg-[#0B9F72] hover:bg-[#168A5B] text-white text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition shadow-xs"
        >
          <Layers className="w-3.5 h-3.5" />
          EXPLORE RECENT MATCHES
        </button>
      </div>

      {/* Footer Info */}
      <div className="mt-8 pt-5 border-t border-[#E3EAF0] text-xs text-[#8A98A8] flex items-center justify-center gap-2">
        <ShieldCheck className="w-3.5 h-3.5 text-[#0B9F72]" />
        <span>T20 120-ball chase model standing by for incoming live state.</span>
      </div>
    </div>
  );
}
