# WANT-8 Vercel and Netlify cross-compatibility

Backend currently targets Netlify Functions exclusively (`netlify/functions/`, `netlify/lib/`). No path to deploying on Vercel without rewriting the serverless layer.

The goal: same codebase builds and deploys on either platform without forking. Differences: Netlify uses `handler(event, context)` + `netlify.toml`; Vercel uses `api/` file-based routing + `vercel.json` + its own request/response types.

Approach: an adapter layer — thin request/response normalisation shim — so business logic in `netlify/lib/` is platform-agnostic and the adapters are the only platform-specific code:
1. Extract handler logic into framework-free modules (most of `netlify/lib/` already qualifies).
2. Write a Netlify adapter and a Vercel adapter (request unwrap + response wrap only).
3. CI builds and lints both targets.

Worth doing when there is a concrete reason to deploy on Vercel — cost comparison, team preference, or a feature only one platform offers.
