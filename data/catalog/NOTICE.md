# Data notice — Gaia DR3 subsets

This directory redistributes small subsets of the European Space Agency's
Gaia DR3 catalogue, queried from the Gaia ESA Archive
(https://gea.esac.esa.int/archive/) via TAP in November 2026.

## Files

| File | Content | Query filter (gaiadr3.gaia_source) |
|---|---|---|
| `part_*.csv` | 225,116 stars, volume-limited (~180 pc) | `parallax > 5.5 AND parallax_over_error > 5 AND phot_g_mean_mag < 12` |
| `bb_*.csv` | 87,434 stars, magnitude-limited, all sky | `phot_g_mean_mag < 8.5` |
| `cat.npz` | 845,673 stars: `u` (ICRS unit direction), `G` (mag), `dpc` (distance pc), `bp` (colour) | `phot_g_mean_mag < 12` with a finite parallax distance (≤ 20 kpc) |

All files are trivial subsets (~0.05% of the 1.8-billion-source DR3
catalogue) with no value added beyond selection and unit conversion.

## Licence and attribution

Gaia data are © ESA/Gaia/DPAC, licensed under the Creative Commons
Attribution-ShareAlike 3.0 IGO licence (CC BY-SA 3.0 IGO), or the ESA
Standard Licence (content can be used under either licence).

If you use these data, please include the standard acknowledgement:

> This work has made use of data from the European Space Agency mission
> Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data
> Processing and Analysis Consortium (DPAC,
> https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the
> DPAC has been provided by national institutions, in particular the
> institutions participating in the Gaia Multilateral Agreement.

Under CC BY-SA 3.0 IGO, any adapted dataset built from these files must be
shared under the same licence. The code in this repository is MIT-licensed
and is not affected by the data licence.
