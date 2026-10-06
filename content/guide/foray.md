# Foray planner

The **Foray** page ranks places to look for what is in season. It also holds
**My areas**, where members record land they may collect on.

> **Note** Fee and collecting information here is an **estimate**. It comes from
> the type of land manager, not from the rules. Rules, permits, quantity limits
> and closures change by unit and by season. Check them with the land manager
> before you go.

## How a place is scored

A place is a grid cell. Its score is the share of the cell's own finds that are
of a species in season for the chosen time. Each species counts in proportion to
how close its usual fruiting is to that day, and a species with a wide fruiting
window counts for a little less.

- The score divides by the cell's own total, so a cell is not better for being
  visited more. It is effort-neutral, like the seasonal heatmap.
- A cell needs **3 or more finds**. Fewer are not shown.
- Only cells that already have finds are scored. A blank area is unsampled, not
  poor. The score does not forecast, and it does not reach beyond sampled cells.

## Choose the time and place

- **When** is *Now* (the last 14 days around today) or a month (about two weeks
  either side of mid-month).
- **Land cover** keeps only finds on one land-cover class.
- **Cell size** sets the grid.

## Three modes

The mode changes the defaults and which panels show. The score is the same.

| Mode | For | Shows |
| --- | --- | --- |
| Forager | A short list in plain terms | Top 8 places with a high, medium or low band |
| Researcher | Checking the numbers | Top 25, score as a percent, the species behind it, sample size, caveats |
| Foray leader | A list to hand out | Top 12 with access, fee and collecting status, and a shortlist. The access switches start on. |

## Access switches

| Switch | Keeps |
| --- | --- |
| Only free places | Cells on land with no fee |
| Only public lands | Cells on land open to the public |
| Collecting allowed only | Cells where collecting is allowed |
| Include likely (BLM/USFS, estimated) | Also counts *likely allowed* land, with the label |
| Keep unknown (Researcher and Foray leader) | Keeps cells where the value is unknown |

Unknown is never assumed free, public or allowed. Unless you keep unknown
places, a switch removes them. A cell that several areas cover takes the most
restrictive value of each.

Access data covers **Colorado only**. Where it is not loaded, the switches turn
off and the page says so. The cell's **centre point** decides its access, so an
edge cell can be labelled by land that covers only part of it.

## The shortlist

The Foray leader mode adds a shortlist with a note field for each place. Notes
stay in your browser. **Export CSV**, **Copy text** and **Print** all carry the
site, coordinates, score, access, fee, collecting, number of finds, top species
and notes. The score is rounded and the fee and collecting columns keep the
*estimated* label.

The dashboard's **Now fruiting** card has a **Plan a foray** link. On the map,
**Heatmap → Foray score** draws the same score as a grid.

## What the labels mean

| Label | Where it comes from |
| --- | --- |
| *estimated* | A rule of thumb from manager type. Not a regulation. |
| *verified* (fee) | Matched to a Recreation.gov facility record. |
| Owner-asserted | Typed by the member who owns the area, in **My areas**. |
| **Likely allowed (estimated)** | Open BLM or US Forest Service land. An estimate, never *allowed*. |

The planner never labels land *allowed* from an estimate. *Allowed* appears
only for areas a member asserted.

## My areas

Members open **My areas** (nav: *Areas*) to keep named sets of places they may
collect. A set is private, or belongs to a **club**.

1. **Create set**, then **Draw polygon** on the map or **Import GeoJSON**
   (Polygon and MultiPolygon features; the import is all or nothing).
2. Name the area, set the **fee** and **collecting** values, and add notes.
   What you enter is **owner-asserted**. It is not verified.
3. For a club, a club owner or admin creates a club set and adds members by
   email. Owners and admins edit the set. Members can view it. A member can also
   edit an area they created.

Your areas draw on the map's **Access** layer as *My areas* and *Club areas*.
Limits: 50 sets each, 500 areas per set, 5,000 vertices per area, 2 MB per
import.

## The Access layer on the map

Open **Layers** and expand **Access**. On a phone it is a section of the layer
sheet. Tick **Show access areas**.

- **Color by** public access, fee status or collecting.
- **Show only** free, public, or collecting-allowed land (and *likely allowed*).
  Unknown areas are hidden by the filters.
- **Sources** toggles *Public lands*, *My areas* and *Club areas*. The last two
  need sign-in. Each shows a count.
- Estimated values draw with a dashed edge, and unknown areas draw grey.
  Clicking an area shows its details with the disclaimer.

Areas load from **zoom 8**. Roads and trails load from **zoom 12**, and only on
a small view. Only Colorado is loaded. Outside it the layer says the data is not
loaded, and no color does not mean no access.

## Not yet verified

The access database, the planner and My areas are built and tested against
fixtures. The access data has not been loaded and checked against live sources.
Treat the first loaded results with caution, and report anything that looks
wrong with [Report a bug](/guide/report-bug).
