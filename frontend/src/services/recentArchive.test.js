import assert from 'node:assert/strict';
import test from 'node:test';
import { loadDemoArchive, loadRecentArchive } from './recentArchive.js';

test('loads stored Recent matches and their detail', async () => {
  const calls = [];
  const archive = await loadRecentArchive(
    'stored-1',
    async () => {
      calls.push('list');
      return { matches: [{ match_id: 'stored-1' }] };
    },
    async (matchId) => {
      calls.push(`detail:${matchId}`);
      return { match_id: matchId, is_recent: true };
    },
  );

  assert.deepEqual(calls, ['list', 'detail:stored-1']);
  assert.equal(archive.selectedMatchId, 'stored-1');
  assert.equal(archive.selectedMatch.is_recent, true);
});

test('loads demo fixtures only through the explicit Demo loader', async () => {
  const calls = [];
  const demo = await loadDemoArchive(
    null,
    async () => {
      calls.push('demo-list');
      return { matches: [{ match_id: 'demo-1', is_demo: true }] };
    },
    async (matchId) => {
      calls.push(`demo-detail:${matchId}`);
      return { match_id: matchId, is_demo: true };
    },
  );

  assert.deepEqual(calls, ['demo-list', 'demo-detail:demo-1']);
  assert.equal(demo.selectedMatch.is_demo, true);
});

test('returns an empty archive without requesting detail', async () => {
  let detailRequested = false;
  const archive = await loadRecentArchive(
    null,
    async () => ({ matches: [] }),
    async () => {
      detailRequested = true;
    },
  );

  assert.deepEqual(archive.matches, []);
  assert.equal(archive.selectedMatchId, null);
  assert.equal(archive.selectedMatch, null);
  assert.equal(detailRequested, false);
});

test('propagates Recent API failures instead of substituting data', async () => {
  await assert.rejects(
    loadRecentArchive(
      null,
      async () => {
        throw new Error('Recent API unavailable');
      },
      async () => {
        throw new Error('Detail should not be requested');
      },
    ),
    /Recent API unavailable/,
  );
});

test('propagates stored-match detail failures without substituting another match', async () => {
  let detailRequests = 0;
  await assert.rejects(
    loadRecentArchive(
      null,
      async () => ({ matches: [{ match_id: 'stored-1' }] }),
      async () => {
        detailRequests += 1;
        throw new Error('Recent detail unavailable');
      },
    ),
    /Recent detail unavailable/,
  );
  assert.equal(detailRequests, 1);
});
