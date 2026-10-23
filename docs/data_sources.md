# Data Sources & Provenance Registry

| Dataset | Source / Provider | Local Path | License | Last Downloaded | Limitations & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Gridded Daily Rainfall (NetCDF)** | IMD Pune (`imdpune.gov.in`) | `data/raw/imdweb_pTxNQ39i.nc` | Open Govt Data | Oct 2026 | 0.25° grid (~27 km resolution); daily aggregations. |
| **IMD District Daily Rainfall (CSV)** | IMD Pune (`imdpune.gov.in`) | `data/raw/rainfall_districtwise_daily_imd.csv` | Open Govt Data | Oct 2026 | District-level average; loses hyper-local variation. |
| **Bengaluru Urban Boundary** | IMD / OpenCity | `data/raw/bengaluru_full_urban_region_imdweb.geojson` | Open Data | Oct 2026 | Spatial boundary polygon for cropping gridded data. |
| **KSNDMC Telemetry Rain Gauges** | KSNDMC via OpenCity (`data.opencity.in`) | To be acquired | Open Data | Pending | Ward-level gauge network (historical years limited). |
| **Google Open Buildings / VIDA Footprints** | Google / Earth Engine | To be acquired | CC BY-4.0 | Pending | Planar footprint area may deviate from actual pitched roof area. |
