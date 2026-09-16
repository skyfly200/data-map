# Analysis

The analysis page computes statistics over the observations that the filters
select.

**Scope** sets whether the page reads everything or only the current filters.
The scope changes each statistic on the page, and also the sample size that the
page reports next to it.

## What relates to what

This tab draws a **Spearman** correlation matrix across each populated field.
Spearman is rank-based, so a relationship counts when it bends. These
relationships bend.

Each cell uses only the rows where *both* fields are present. That matters here.
Soil moisture is on approximately 23% of the rows. If the app removed each row
that is missing any field, it would compute the whole matrix on a remainder that
is not representative.

> **Caution** Two confounds run through each pair.
> **Season**: high-elevation finds happen in summer, and low ones happen in
> spring and autumn. Elevation and temperature therefore appear to increase
> together (ρ +0.43). Hold the month still and that falls to approximately zero.
> **Effort**: people record where people go.

## Species

### Fingerprints

A fingerprint shows how far one species is from the dataset average on each
field. The unit is standard deviations. A positive value means higher, warmer,
wetter or later than the average.

The app uses z-scores and not raw means. You cannot compare metres, degrees and
millimetres side by side.

### Found together

This chart ranks pairs of species by **lift**. Lift is how much more often two
species occur together than their individual frequencies predict. The app counts
a co-occurrence inside the same cell of approximately 5 km and the same month.

Lift, and not a raw count. A count would rank the two most common species first,
whether or not they have any relation to each other.

## Fruiting timing

This tab shows what moves the fruiting of a species earlier or later from year
to year. It also shows whether that movement follows rainfall, temperature, or
both.

Timing is the median day of the year. The app ranks the drivers by partial
correlation, so a driver that looks significant only because it follows another
driver goes down the list.

The rainfall chart shows one year against the average, and against the previous
years. The window is 30 days, trailing.

> **Note** The app offers only the species with at least 20 observations across
> four or more complete years. Below that, the year-to-year signal is noise.

## Year over year

This tab shows the season timing (the median day of the year) and the median
elevation for each year. It shows the recording effort next to them, because
effort drives both.

The app uses the median and not the mean. A few winter records would move a mean
a long way.

> **Caution** Read the effort chart first. Recording has increased steeply. A
> change in either trend can be a change in who is looking.

## Data quality

This tab shows the field coverage overall and by year. It tells you how much
weight the rest of the app can carry.

A chart drawn from a column that is 23% complete looks exactly as confident as a
chart drawn from a full column.

A field is thin because the enrichment has not reached those rows. Run the
pipeline again to fill them. Read [Pipeline jobs](/guide/jobs).
