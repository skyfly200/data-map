# Pipeline jobs

An FRMS member can run the enrichment pipeline from the app. You do not need a
Python environment. You do not need Earth Engine credentials of your own.

Open **Jobs** in the header, or **Pipeline jobs** in the account menu.

## How to run a job

1. Give the job a name.
2. Select what to run it over. Read [Choose a source](#choose-a-source).
3. Select the layers to sample.
4. Queue the job.

The page shows the progress against the job while it runs. You can leave the
page. The job continues.

When the job finishes, **Open on map** loads the result as a dataset. The map,
the charts and the analysis then all read it.

**Export** downloads the result directly, as GeoJSON or CSV. Read
[Export](/guide/data#export).

## Choose a source

A job runs over one of two sources.

**Observations in an area** uses the iNaturalist records inside a bounding box.
Enter the box, or use the current map view. You can also set a date range and a
taxon.

**A saved dataset** uses the points of a dataset that you saved before. The app
uses the points and the dates of that dataset as they are. The area, the dates
and the taxon do not apply, because those chose the points in the first place.

## Save a result as a dataset

A finished job has a **Save as dataset** button. Give the dataset a name and
select who can use it.

| Visibility | Who can read it |
| --- | --- |
| Only me | You, and nobody else |
| FRMS members | Any signed-in member |

New datasets are **private by default**. You must change the visibility
yourself. You can change it again at any time from **Your datasets**.

A dataset that you cannot read is indistinguishable from a dataset that does not
exist. The app returns the same answer for both, so the list of names that other
people use stays private.

Each member can hold up to 100 datasets.

## Chain jobs together

A saved dataset can be the source of the next job. This is what makes jobs
compose.

![Enrich a set of points, save the result, then enrich the result again.](figure:job-chain)

Run a job over an area with six layers. Save the result. Run a second job over
that dataset with four more layers. Save the wider result. Repeat.

This is useful for three reasons:

- A job with every layer is slow and expensive. A job with six layers is not.
- You can add a layer to a set of points months later, without a new fetch of
  the points.
- The points stay the same across the chain, so each column describes the same
  records.

## Budgets

> **Note** Earth Engine bills the project and not the caller. Each job spends
> from one shared pool.

An administrator sets four limits for each member:

- A monthly budget in Earth Engine units.
- A number of jobs for each day.
- A maximum number of points for each job.
- A maximum number of jobs that run at the same time.

Your current limits are on the jobs page.
