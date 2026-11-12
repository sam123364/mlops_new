# Data Sources & Provenance Registry

| Dataset | Source / Provider | Local Path | License | Last Downloaded | Limitations & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Gridded Daily Rainfall (NetCDF)** | IMD Pune (`imdpune.gov.in`) | `data/raw/imdweb_pTxNQ39i.nc` | Open Govt Data | Oct 2026 | 0.25° grid (~27 km resolution); daily aggregations. |
| **IMD District Daily Rainfall (CSV)** | IMD Pune (`imdpune.gov.in`) | `data/raw/rainfall_districtwise_daily_imd.csv` | Open Govt Data | Oct 2026 | District-level average; loses hyper-local variation. |
| **Bengaluru Urban Boundary** | IMD / OpenCity | `data/raw/bengaluru_full_urban_region_imdweb.geojson` | Open Data | Oct 2026 | Spatial boundary polygon for cropping gridded data. |
| **Niño 3.4 SST anomaly (monthly)** | NOAA PSL, ERSST v6, `https://psl.noaa.gov/data/correlation/nina34.anom.data` | `data/raw/climate_indices/nina34.anom.data` (DVC) | NOAA data; licence not yet verified | 2026-11-12 | 1948–2026; anomaly vs 1981–2010; reanalysed (not real-time); missing code -99.99. |
| **Dipole Mode Index, IOD (monthly)** | NOAA PSL from HadISST1.1, `https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data` | `data/raw/climate_indices/dmi.had.long.data` (DVC) | HadISST (Met Office) terms; licence not yet verified | 2026-11-12 | 1870–2026; file marked "Preliminary"; missing code -9999. |
| **KSNDMC Telemetry Rain Gauges** | KSNDMC via OpenCity (`data.opencity.in`) | To be acquired | Open Data | Pending | Ward-level gauge network (historical years limited). |
| **Google Open Buildings / VIDA Footprints** | Google / Earth Engine | To be acquired | CC BY-4.0 | Pending | Planar footprint area may deviate from actual pitched roof area. |
