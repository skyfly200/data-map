# WANT-3 Member-supplied Earth Engine credentials

Every job runs under one service account against one Cloud project, so all Earth Engine spend bills FRMS. That single pool is the whole reason `quotas.mjs` exists: the monthly units, the per-day and concurrency caps, and the admin floor all divide one budget fairly rather than protect against any one member. A member who needs more than their share, or who wants to run heavier processing than the shared budget should carry, has no answer today but "ask an admin to raise the number".

The way out is to let a member run their own jobs under their own Earth Engine project. `initEarthEngine` already takes a service-account key from the environment; the change is to take the job owner's stored credential instead when they have one, falling back to the shared account when they do not. A job on a member's own project spends the member's own Earth Engine budget, so `checkQuota` skips the shared monthly cap for it — they are metered by Google, not by the pool — while the point and concurrency caps stay.

The hard part is holding the credential. A Google service-account key is a long-lived secret. Options:
- Service-role-only column encrypted at rest with a key not in the same table, with validation round-trip and self-service rotation/revoke.
- OAuth — member authorises Nexstrata against their EE project, app holds a refresh token. More to build but safer posture.

Either way the credential never reaches the browser and never appears in a job spec. The guide already covers creating a Cloud project and registering it for EE (`guide/layers`).

Worth doing when a member's needs exceed what the shared budget should fund — the first real "I need my own quota" is the signal.
