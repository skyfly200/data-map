# Data and export

## Species

The species list shows each species with its record count. You can group the
list at genus, species or subspecies level. When you select an entry, each view
narrows at the same time.

The **Rank** selector sets the level of the list and the level of the filter
together. Read [Taxonomy](/guide/map#taxonomy) for what each rank contains.

## Table

The table shows all the observations with their enriched columns. You can sort
it and search it. The app renders only the visible rows, so the full set stays
responsive.

## Filters

A filter applies everywhere: to the map, the charts, the table and the analysis.

| Filter | Notes |
| --- | --- |
| Country, state, county | Parsed from the iNaturalist place string |
| Radius | The distance from a point that you choose |
| Year, month, week | ISO week, to match the week-of-year chart |
| Date range | A start date and an end date |
| Minimum records | Removes taxa below a count, by species or by genus |

**Minimum records** counts the records *after* the other filters. It therefore
means "enough records in what I am looking at". When you narrow to one county
and ask for 25, you get the species that are well sampled in that county.

A species that was seen twice cannot tell you where it fruits.

**Saved subsets** store the full filter state under a name. One click restores
it. The chip describes itself from what is actually set.

## Export

The **Export** button is on the table and on each finished job. It downloads the
records that you can see, with the filters applied.

### Choose a format

- **GeoJSON** puts each record in the geometry, with its properties. Use it for
  GIS software: QGIS, ArcGIS, geopandas.
- **CSV** writes one row for each record. Use it for a spreadsheet.

### Choose the columns

An enriched row is wide. A job that ran each stage produces approximately forty
columns. The menu therefore lets you choose:

- **Shown in the table** exports the columns that are on the screen. This is the
  default.
- **Everything** exports each column in the dataset.
- **Choose** opens a checkbox for each column.

The coordinates always survive. In GeoJSON they are in the geometry, so a
narrow column list cannot remove them. In CSV they get their own longitude and
latitude columns, because a CSV has no other place for them.

> **Note** Some sources fetch their rows only when you download them. The app
> cannot list the columns of one of these before the download, so it offers no
> column picker and exports everything.

### What the file contains

- The CSV uses `\r\n` line endings and a UTF-8 byte-order mark. Excel then reads
  accented names correctly.
- A value that starts with `=`, `+`, `-` or `@` gets a leading apostrophe. A
  spreadsheet reads these characters as the start of a formula. Without the
  apostrophe, a species name, or a value from a record that somebody else wrote,
  could run as code when you open the file.
- The file name carries the source and the date, for example
  `observations-2026-09-16.csv`.

> **Caution** An export contains the coordinates of each record, and some of
> those coordinates are obscured. Read the terms of the source before you
> republish an export. iNaturalist records carry their own licences.
