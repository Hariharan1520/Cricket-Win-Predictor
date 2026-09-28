import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import MatchSelector from './components/MatchSelector';
import MatchHeader from './components/MatchHeader';
import WinProbabilityCard from './components/WinProbabilityCard';
import MatchStateCard from './components/MatchStateCard';
import ProbabilityTimeline from './components/ProbabilityTimeline';
import ProbabilitySwings from './components/ProbabilitySwings';
import NextOverSimulator from './components/NextOverSimulator';
import ModelDetailsCard from './components/ModelDetailsCard';
import NoLiveMatchState from './components/NoLiveMatchState';
import {
  fetchLiveMatches,
  fetchMatchDetail,
  fetchRecentMatches,
  fetchRecentMatchDetail,
  fetchDemoMatches,
  fetchDemoMatchDetail,
  fetchHealth,
} from './services/api';
import { ShieldCheck, AlertCircle } from 'lucide-react';

const DEFAULT_POLL_INTERVAL = 30; // seconds

export default function App() {
  const [activeMode, setActiveMode] = useState('live'); // 'live' | 'recent'
  const [liveMatches, setLiveMatches] = useState([]);
  const [recentMatches, setRecentMatches] = useState([]);
  const [selectedMatchId, setSelectedMatchId] = useState(null);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [pollCountdown, setPollCountdown] = useState(DEFAULT_POLL_INTERVAL);
  const [backendHealth, setBackendHealth] = useState(null);
  const [activeNav, setActiveNav] = useState('live-matches');

  // Check backend health on mount
  useEffect(() => {
    fetchHealth()
      .then((data) => setBackendHealth(data))
      .catch((err) => {
        console.error('Backend health check error:', err);
        setError('Cannot reach Flask backend server. Ensure backend/app.py is running on port 5000.');
      });
  }, []);

  // Fetch match lists when mode changes
  const loadMatches = useCallback(async (isBackground = false) => {
    if (!isBackground) setIsLoading(true);
    setIsRefreshing(true);
    setError(null);

    try {
      if (activeMode === 'live') {
        const data = await fetchLiveMatches();
        const matches = data.matches || [];
        setLiveMatches(matches);

        if (matches.length > 0) {
          const nextId = matches.some((m) => m.match_id === selectedMatchId)
            ? selectedMatchId
            : matches[0].match_id;
          setSelectedMatchId(nextId);
          const detail = await fetchMatchDetail(nextId);
          setSelectedMatch(detail);
        } else {
          setSelectedMatchId(null);
          setSelectedMatch(null);
        }
      } else {
        // Recent matches mode (backed by persistent DB)
        let matches = [];
        try {
          const data = await fetchRecentMatches();
          matches = data.matches || [];
        } catch (e) {
          console.warn('Recent matches endpoint failed, attempting fallback to demo matches:', e);
        }

        if (matches.length === 0) {
          // Fallback to demo matches if database has not yet been populated
          const demoData = await fetchDemoMatches();
          matches = demoData.matches || [];
        }

        setRecentMatches(matches);

        const nextId = matches.some((m) => m.match_id === selectedMatchId)
          ? selectedMatchId
          : matches[0]?.match_id;
        setSelectedMatchId(nextId);
        if (nextId) {
          try {
            const detail = await fetchRecentMatchDetail(nextId);
            setSelectedMatch(detail);
          } catch (e) {
            const demoDetail = await fetchDemoMatchDetail(nextId);
            setSelectedMatch(demoDetail);
          }
        }
      }
    } catch (err) {
      console.error('Error loading matches:', err);
      setError(err.message || 'Failed to communicate with backend server.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
      setPollCountdown(DEFAULT_POLL_INTERVAL);
    }
  }, [activeMode, selectedMatchId]);

  // Handle match selection
  const handleSelectMatch = async (matchId) => {
    setSelectedMatchId(matchId);
    setIsRefreshing(true);
    try {
      if (activeMode === 'live') {
        const detail = await fetchMatchDetail(matchId);
        setSelectedMatch(detail);
      } else {
        try {
          const detail = await fetchRecentMatchDetail(matchId);
          setSelectedMatch(detail);
        } catch (e) {
          const demoDetail = await fetchDemoMatchDetail(matchId);
          setSelectedMatch(demoDetail);
        }
      }
    } catch (err) {
      console.error('Error fetching match detail:', err);
      setError(err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Initial load on mode switch
  useEffect(() => {
    loadMatches();
  }, [activeMode]);

  // Polling timer (30s)
  useEffect(() => {
    const timer = setInterval(() => {
      setPollCountdown((prev) => {
        if (prev <= 1) {
          loadMatches(true);
          return DEFAULT_POLL_INTERVAL;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [loadMatches]);

  const handleNavClick = (navId) => {
    setActiveNav(navId);
    const element = document.getElementById(navId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const currentMatchesList = activeMode === 'live' ? liveMatches : recentMatches;
  const isCurrentlyLive = activeMode === 'live' && liveMatches.length > 0;
  const isRecent = activeMode === 'recent' || activeMode === 'demo';

  return (
    <div className="min-h-screen bg-[#F5F8FB] text-[#172B4D] flex flex-col font-sans selection:bg-[#0B9F72] selection:text-white">
      {/* Top Horizontal Header */}
      <Header
        isLive={isCurrentlyLive}
        isDemo={isRecent}
        onRefresh={() => loadMatches(false)}
        isRefreshing={isRefreshing}
        activeMode={activeMode}
        setActiveMode={(mode) => {
          setActiveMode(mode);
          setError(null);
        }}
        pollCountdown={pollCountdown}
        activeNav={activeNav}
        onNavClick={handleNavClick}
      />

      {/* Horizontal Live / Recent Match Strip */}
      {currentMatchesList.length > 0 && (
        <MatchSelector
          matches={currentMatchesList}
          selectedMatchId={selectedMatchId}
          onSelectMatch={handleSelectMatch}
          title={activeMode === 'live' ? 'Live Matches' : 'Recent Matches'}
          isDemo={isRecent}
        />
      )}

      {/* Main Analytics Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-5">
        {/* Error Banner */}
        {error && (
          <div className="mb-5 p-3.5 rounded-xl bg-[#FFF0F2] border border-[#EF5B67]/30 text-[#EF5B67] text-xs font-semibold flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Refined Persistent Recent Match Archive Banner */}
        {isRecent && (
          <div className="mb-5 px-3.5 py-2 rounded-xl bg-[#F0F4F8] border border-[#D9E2EC] text-[#334E68] text-xs flex items-center justify-between gap-3 shadow-2xs">
            <div className="flex items-center gap-2 text-xs">
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase bg-[#D9E2EC] text-[#102A43] tracking-wider">
                RECENT MATCH REPLAY
              </span>
              <span>
                Persistent PostgreSQL archive · Ball-by-ball win-probability trajectory & swings.
              </span>
            </div>
            <button
              onClick={() => setActiveMode('live')}
              className="px-2.5 py-1 bg-white hover:bg-[#F5F8FB] text-[#102A43] border border-[#CBD2D9] rounded text-[11px] font-bold uppercase transition flex-shrink-0 shadow-2xs"
            >
              Switch to Live API
            </button>
          </div>
        )}

        {/* Live Mode & No T20 Available */}
        {activeMode === 'live' && liveMatches.length === 0 && !isLoading && (
          <NoLiveMatchState
            onSwitchToRecent={() => setActiveMode('recent')}
            onSwitchToDemo={() => setActiveMode('recent')}
            onRefresh={() => loadMatches(false)}
            isRefreshing={isRefreshing}
          />
        )}

        {/* Selected Match Details & Analytics */}
        {selectedMatch && selectedMatch.available && (
          <div className="space-y-5">
            {/* 1. Match Hero / Scoreboard */}
            <MatchHeader match={selectedMatch} />

            {/* 2. Win Probability Hero */}
            <WinProbabilityCard match={selectedMatch} />

            {/* 3. Match State (Single Horizontal Card + 8 Features Drawer) */}
            <MatchStateCard match={selectedMatch} />

            {/* 4. Analytics Row: Timeline (Left) + Swings (Right) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-7">
                <ProbabilityTimeline
                  timeline={selectedMatch.timeline}
                  chasingTeam={selectedMatch.chasing_team}
                  defendingTeam={selectedMatch.defending_team}
                />
              </div>
              <div className="lg:col-span-5">
                <ProbabilitySwings recentSwings={selectedMatch.recent_swings} />
              </div>
            </div>

            {/* 5. Simulation & Model Specifications Row */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-8">
                <NextOverSimulator
                  scenarios={selectedMatch.scenarios}
                  currentWinProbPct={selectedMatch.win_probability_pct}
                />
              </div>
              <div className="lg:col-span-4">
                <ModelDetailsCard />
              </div>
            </div>
          </div>
        )}

        {/* Unavailable Match State (e.g. non-T20, 1st innings, or provider without ball-by-ball) */}
        {selectedMatch && !selectedMatch.available && (
          <div className="p-8 rounded-2xl bg-white border border-[#E3EAF0] text-center max-w-xl mx-auto my-12 shadow-xs">
            <h3 className="text-base font-bold text-[#172B4D] mb-1.5">
              {selectedMatch.match_name || selectedMatch.match?.name || 'Selected Match'}
            </h3>
            <p className="text-xs text-[#667085] mb-4">
              {selectedMatch.reason || 'Historical probability replay unavailable for this match because ball-by-ball data was not provided by the data source.'}
            </p>
            <div className="inline-flex items-center gap-2 text-xs px-3 py-1 bg-[#F5F8FB] text-[#667085] rounded-full border border-[#E3EAF0]">
              <span>Status: {selectedMatch.status || selectedMatch.match?.status || 'In Progress'}</span>
              {(selectedMatch.venue || selectedMatch.match?.venue) && (
                <span>· {selectedMatch.venue || selectedMatch.match?.venue}</span>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Clean Light Footer */}
      <footer className="border-t border-[#E3EAF0] bg-white px-4 sm:px-6 py-3.5 text-xs text-[#667085] mt-6">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-[#0B9F72]" />
            <span>Cricket Win Predictor · Professional T20 Analytics Platform</span>
          </div>
          <div className="text-[11px] text-[#8A98A8]">
            Phase 3 Logistic Regression Baseline · 120-Ball Second-Innings Model
          </div>
        </div>
      </footer>
    </div>
  );
}
