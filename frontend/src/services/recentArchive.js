async function loadMatchArchive(selectedMatchId, fetchMatches, fetchMatchDetail) {
  const response = await fetchMatches();
  const matches = response.matches || [];
  const nextMatchId = matches.some((match) => match.match_id === selectedMatchId)
    ? selectedMatchId
    : matches[0]?.match_id ?? null;
  const selectedMatch = nextMatchId
    ? await fetchMatchDetail(nextMatchId)
    : null;

  return {
    matches,
    selectedMatchId: nextMatchId,
    selectedMatch,
  };
}

export const loadDemoArchive = loadMatchArchive;
export const loadRecentArchive = loadMatchArchive;
