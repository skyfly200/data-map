# WANT-8 Vercel and Netlify cross-compatibility

Backend currently targets Netlify Functions exclusively (`netlify/functions/`, `netlify/lib/`). No path to deploying on Vercel without rewriting the serverless layer.

The goal: same codebase builds and deploys on either platform without forking. Differences: Netlify uses `handler(event, context)` + `netlify.toml`; Vercel uses `api/` file-based routing + `vercel.json` + its own request/response types.

Approach: an adapter layer — thin request/response normalisation shim — so business logic in `netlify/lib/` is platform-agnostic and the adapters are the only platform-specific code:
1. Extract handler logic into framework-free modules (most of `netlify/lib/` already qualifies).
2. Write a Netlify adapter and a Vercel adapter (request unwrap + response wrap only).
3. CI builds and lints both targets.

Worth doing when there is a concrete reason to deploy on Vercel — cost comparison, team preference, or a feature only one platform offers.

## Status

Done (unverified on Vercel):
- `netlify/lib/storage.mjs` replaces direct Blobs use; `STORAGE_BACKEND=supabase|netlify` (default by `NETLIFY`), private bucket objects at `<store>/<key>`.
- `nuxt.config.ts` picks the `vercel` preset when `VERCEL` is set; `server/adapters/vercel-function.ts` serves `netlify/functions/<name>.mjs` at `/api/fn/<name>`; `scripts/gen-vercel-config.mjs` writes `vercel.json` (rewrite plus crons).

Not done: any deploy on Vercel, a CI build of both targets. Env vars: `docs/deploying.md` block 7.
