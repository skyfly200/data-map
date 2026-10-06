# Access, fee & collecting sources (Colorado)

**Regulations must be checked locally.** Nothing in the app is a legal statement. Estimates are labelled `estimated`; `likely_allowed` (BLM/USFS only) is a guess, and the app never shows plain `allowed` from an estimate. Only member-entered areas are `allowed`, owner-asserted.

Where to look up land manager, fee and mushroom-collecting rules for the Foray planner (WANT-17). The app's `fee_status` / `collecting` values are **estimates** from manager type; these pages are the authority. Checked 2026-10; re-verify before relying on any rule. Pages marked (secondary) are blogs/news, not rule text.

## Boundaries & manager type (what we ingest)

| Source | Covers | Notes |
|---|---|---|
| [USGS PAD-US](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-overview) | Federal, state, local, some private conservation lands; manager, designation, public access | Primary ingest source (`access_areas`; URL and field names unverified, see `docs/deploying.md` block 7). State and many county/city parks appear as manager type STAT/LOC. |
| [COMaP, Colorado Ownership, Management & Protection](https://comap.cnhp.colostate.edu/) (CNHP) | ~28k protected-land polygons from 300+ sources, incl. county/city open space and conservation easements | Free, registration required. Best Colorado supplement to PAD-US for county/city parks. See [CNHP COMaP](https://cnhp.colostate.edu/projects/comap). |
| [Colorado Geospatial Portal](https://geodata.colorado.gov/) | State GIS hub, incl. [CPW administrative boundaries](https://geodata.colorado.gov/content/0c363dc22a4a4d11b8c157015de8b704) | Authoritative for state parks/wildlife areas. |
| [Boulder County open data](https://bouldercounty.gov/open-space/maps/) | Example county portal (open space, trails) | Counties publish their own; add per county as needed. |
| [OpenStreetMap / Overpass](https://overpass-turbo.eu/) | Roads, trails | Already ingested (`access_lines`). |

## Fees

- **State parks:** all Colorado state parks charge entrance fees; vehicle pass required. [CPW parks passes](https://cpw.state.co.us/parks-passes), [specialty passes](https://cpw.state.co.us/park-specialty-passes). Fee levels change (non-resident surcharge: (secondary) [CPR, 2026-05](https://www.cpr.org/2026/05/04/colorado-state-parks-charge-out-of-state-visitors/)), so we store fee *status*, not amounts.
- **Federal sites (USFS/BLM/NPS):** [Recreation.gov RIDB](https://ridb.recreation.gov/) API, `RIDB_API_KEY` overlay, implemented but not run live (`fee_source='ridb'`; field names unverified).
- **County/city open space:** most are free to enter but vary; check the owning agency's site (no statewide dataset known). Left `unknown`.

## Collecting rules (what to check, by manager)

- **USFS (national forests):** personal-use mushroom rules are set per forest/district: some need a free permit and quantity limits (limits quoted for White River NF: 5 gal/day, 67 lb/season, permit from Dillon Ranger District; incidental camp use exempt; commercial needs paid permit). (secondary) [Summit Daily](https://www.summitdaily.com/news/dont-forget-your-permit-when-foraging-fungi-in-the-national-forest/). Confirm with the local ranger district (find it via [fs.usda.gov](https://www.fs.usda.gov/)) before going. Tag: `likely_allowed` (estimated).
- **BLM:** personal-use amounts are reported as permit-free in Colorado, but confirm with the field office. (secondary) [Modern Forager](https://modern-forager.com/colorado-public-lands-mushroom-foraging/); BLM general guidance: [Can I Keep This?](https://www.blm.gov/Learn/Can-I-Keep-This). Tag: `likely_allowed` (estimated).
- **Colorado state parks (CPW):** state rules prohibit removing vegetation on Parks and Outdoor Recreation lands ([2 CCR 405-1](https://www.sos.state.co.us/CCR/GenerateRulePdf.do?ruleVersionId=2387&fileName=2+CCR+405-1)). Secondary sources report most parks ban collecting, with a free permit at some (Golden Gate Canyon, Castlewood Canyon); **unverified**, so call the park office. Tag: `restricted`.
- **County open space:** typically prohibits taking vegetation. Example: Jefferson County Open Space [regulations](https://www.jeffco.us/1583/Regulations) ($100 fine; scientific collecting needs a Research & Collections permit). Boulder County/City of Boulder are similarly restrictive (secondary). Check each agency's rules page. Tag: `restricted`/`unknown`.
- **National parks (NPS):** check park-specific compendium. Tag: `restricted`.
- **Wilderness, closures:** layer extra rules on top of manager rules. Tag: `restricted`.

(secondary roundups: [Modern Forager](https://modern-forager.com/colorado-public-lands-mushroom-foraging/), [Mushroom Tracker](https://www.mushroomtracker.ca/blog/mushroom-foraging-colorado.html))

## Gaps / next steps

- No statewide dataset of county/city collecting rules exists; options are a curated table keyed by managing agency (admin-editable) with a source URL and `verified_on` date.
- Ingest COMaP (needs registration) to improve county/city coverage beyond PAD-US.
- Verify the state-park permit exceptions directly with CPW.
