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
import { loadDemoArchive, loadRecentArchive } from './services/recentArchive';
import { ShieldCheck, AlertCircle } from 'lucide-react';

const DEFAULT_POLL_INTERVAL = 30; // seconds — mutable via Settings

export default function App() {
  const [activeMode, setActiveMode] = useState('live'); // 'live' | 'recent'
  const [liveMatches, setLiveMatches] = useState([]);
  const [recentMatches, setRecentMatches] = useState([]);
  const [demoMatches, setDemoMatches] = useState([]);
  const [selectedMatchId, setSelectedMatchId] = useState(null);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [pollInterval, setPollInterval] = useState(DEFAULT_POLL_INTERVAL); // user-configurable via Settings
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
      } else if (activeMode === 'demo') {
        const demo = await loadDemoArchive(
          selectedMatchId,
          fetchDemoMatches,
          fetchDemoMatchDetail,
        );
        setDemoMatches(demo.matches);
        setSelectedMatchId(demo.selectedMatchId);
        setSelectedMatch(demo.selectedMatch);
      } else {
        const archive = await loadRecentArchive(
          selectedMatchId,
          fetchRecentMatches,
          fetchRecentMatchDetail,
        );
        setRecentMatches(archive.matches);
        setSelectedMatchId(archive.selectedMatchId);
        setSelectedMatch(archive.selectedMatch);
      }
    } catch (err) {
      console.error('Error loading matches:', err);
      if (activeMode === 'recent') {
        setRecentMatches([]);
        setSelectedMatchId(null);
        setSelectedMatch(null);
        setError('Unable to load recent matches.');
      } else {
        setError(err.message || 'Failed to communicate with backend server.');
      }
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
      setPollCountdown(pollInterval);
    }
  }, [activeMode, selectedMatchId, pollInterval]);

  // Handle match selection
  const handleSelectMatch = async (matchId) => {
    setSelectedMatchId(matchId);
    setIsRefreshing(true);
    try {
      if (activeMode === 'live') {
        const detail = await fetchMatchDetail(matchId);
        setSelectedMatch(detail);
      } else if (activeMode === 'demo') {
        const detail = await fetchDemoMatchDetail(matchId);
        setSelectedMatch(detail);
      } else {
        setSelectedMatch(null);
        const detail = await fetchRecentMatchDetail(matchId);
        setSelectedMatch(detail);
      }
    } catch (err) {
      console.error('Error fetching match detail:', err);
      setError(activeMode === 'recent' ? 'Unable to load recent matches.' : err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Initial load on mode switch
  useEffect(() => {
    loadMatches();
  }, [activeMode]);

  // Polling timer — interval configurable via Settings, active ONLY in Live mode
  useEffect(() => {
    if (activeMode !== 'live') return;
    const timer = setInterval(() => {
      setPollCountdown((prev) => {
        if (prev <= 1) {
          loadMatches(true);
          return pollInterval;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [loadMatches, pollInterval, activeMode]);

  const handleNavClick = (navId) => {
    setActiveNav(navId);
    const element = document.getElementById(navId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const currentMatchesList = activeMode === 'live'
    ? liveMatches
    : activeMode === 'demo'
      ? demoMatches
      : recentMatches;
  const isCurrentlyLive = activeMode === 'live' && liveMatches.length > 0;
  const isRecent = activeMode === 'recent';
  const isDemo = activeMode === 'demo';

  return (
    <div className="min-h-screen bg-[#F5F8FB] text-[#172B4D] flex flex-col font-sans selection:bg-[#0B9F72] selection:text-white">
      {/* Top Horizontal Header */}
      <Header
        isLive={isCurrentlyLive}
        onRefresh={() => loadMatches(false)}
        isRefreshing={isRefreshing}
        activeMode={activeMode}
        setActiveMode={(mode) => {
          setActiveMode(mode);
          setError(null);
        }}
        pollInterval={pollInterval}
        onPollIntervalChange={(newInterval) => {
          setPollInterval(newInterval);
          setPollCountdown(newInterval); // reset countdown immediately
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
          title={activeMode === 'live' ? 'Live Matches' : isDemo ? 'Demo Matches' : 'Recent Matches'}
          isDemo={isDemo}
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

        {/* Accurate Persistent Recent Match Archive Banner */}
        {activeMode === 'recent' && (
          <div className="mb-5 px-3.5 py-2 rounded-xl bg-[#F0F4F8] border border-[#D9E2EC] text-[#334E68] text-xs flex items-center justify-between gap-3 shadow-2xs">
            <div className="flex items-center gap-2 text-xs">
              <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase bg-[#D9E2EC] text-[#102A43] tracking-wider">
                RECENT MATCH ARCHIVE
              </span>
              <span>
                Persistent PostgreSQL storage · Real completed T20 matches
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

        {activeMode === 'recent' && !isLoading && !error && recentMatches.length === 0 && (
          <div
            role="status"
            className="my-12 rounded-xl border border-[#D9E2EC] bg-white px-6 py-8 text-center text-sm font-semibold text-[#667085]"
          >
            No recent matches are currently archived.
          </div>
        )}

        {/* Live Mode & No T20 Available */}
        {activeMode === 'live' && liveMatches.length === 0 && !isLoading && (
          <NoLiveMatchState
            onSwitchToRecent={() => setActiveMode('recent')}
            onSwitchToDemo={() => setActiveMode('demo')}
            onRefresh={() => loadMatches(false)}
            isRefreshing={isRefreshing}
          />
        )}

        {/* Selected Match Details & Analytics */}
        {selectedMatch && selectedMatch.available && (
          <div className="space-y-5">
            {/* 1. Match Hero / Scoreboard */}
            <MatchHeader match={selectedMatch} isRecent={isRecent} isDemo={isDemo} />

            {/* 2. Win Probability Hero */}
            <WinProbabilityCard match={selectedMatch} isRecent={isRecent} isDemo={isDemo} />

            {/* 3. Match State (Single Horizontal Card + 8 Features Drawer) */}
            <MatchStateCard match={selectedMatch} />

            {/* 4. Analytics Row: Timeline (Left) + Swings (Right) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-7">
                <ProbabilityTimeline
                  timeline={selectedMatch.timeline}
                  chasingTeam={selectedMatch.chasing_team}
                  defendingTeam={selectedMatch.defending_team}
                  isRecent={isRecent}
                />
              </div>
              <div className="lg:col-span-5">
                <ProbabilitySwings
                  recentSwings={isRecent && selectedMatch.significant_swing_events?.length > 0 ? selectedMatch.significant_swing_events : selectedMatch.recent_swings}
                  isRecent={isRecent}
                  chasingTeam={selectedMatch.chasing_team}
                  defendingTeam={selectedMatch.defending_team}
                />
              </div>
            </div>

            {/* 5. Simulation & Model Specifications Row */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              <div className="lg:col-span-8">
                <NextOverSimulator
                  scenarios={selectedMatch.scenarios}
                  currentWinProbPct={selectedMatch.win_probability_pct}
                  isRecent={isRecent}
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
