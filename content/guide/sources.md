# Where the data comes from

The observations come from iNaturalist. Each environmental column is sampled
from Google Earth Engine at the coordinate of the observation, on the date of
the observation.

| Column | Source |
| --- | --- |
| `ndvi` | Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`) |
| `soil_moisture` | ERA5-Land daily |
| `prcp_d0..d6` | CHIRPS daily rainfall |
| `tmax_d0..d6`, `tmin_d0..d6` | ERA5-Land daily |
| `wind_u`, `wind_v` | ERA5-Land 10 m wind |
| `land_cover` | ESA WorldCover |
| `elevation`, `slope`, `aspect` | SRTM |
| Solar exposure, wind exposure, wetness index | Derived from the terrain, plus MERIT Hydro |

The app stores the wind as vector components and not as a bearing. Directions
are circular: the average of 350° and 10° is 180°, which points in exactly the
wrong direction.

## What a sampled value means

The pipeline reads the raster cell that contains the point. It does not
interpolate between cells, and it does not use a regional mean.

The native resolutions run from 10 m to approximately 10 km. Read [Resolution
and scale](/guide/learning#resolution-and-scale) for what that does to a value.

> **Caution** A record whose coordinates are obscured or coarse is flagged. The
> terrain that is sampled at an obscured point describes a place where the
> observation was not.

## Sharing and saving

**Share** builds a link that reproduces the current view. The map position, the
filters, the coloring, the heatmap and the date window all travel with it.

From the same menu you can produce a QR code, or share to X, Bluesky, Facebook,
Reddit, email or SMS. You can also copy an iframe snippet that embeds the view
without the site header.

The app generates the QR code in the browser. The link therefore never reaches a
third party.

**Save image** exports a chart as a PNG or an SVG. It exports the map as a PNG,
with the basemap composited into it.

**Accounts** are optional. When you are signed out, everything persists in this
browser. When you sign in, your settings and your saved charts follow you across
devices.

The first sign-in on a device merges your local work into your account. It does
not overwrite it, so work that you did before you made the account is not lost.

## Terms

Each source sets its own terms. Read them before you republish anything.

- [iNaturalist terms of
  use](https://www.inaturalist.org/pages/terms) — each observation also carries
  its own licence.
- [The Earth Engine data
  catalogue](https://developers.google.com/earth-engine/datasets) — each dataset
  page names the terms of that dataset.
- [Nexstrata privacy policy](/privacy) and [terms](/terms).
