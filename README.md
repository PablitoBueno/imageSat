
# Abstract

This technical article presents a comprehensive workflow for extracting and analyzing the Normalized Difference Vegetation Index (NDVI) from Sentinel-2 imagery over a user-defined Area of Interest (AOI). Leveraging Google Earth Engine and open-source Python libraries (geemap, Prophet), the pipeline enables interactive AOI selection, monthly NDVI time-series extraction, six-month forecasting, multi-index computation, and risk flagging. The approach is validated through a case study and its reproducibility is ensured via detailed methodological descriptions.

# 1. Introduction

Monitoring vegetation dynamics is critical for environmental management, agriculture, and climate studies. NDVI, a widely used index derived from red and near-infrared bands, provides insights into plant health and biomass. Traditional methods for NDVI analysis often require manual data handling and specialized software. This work integrates Google Earth Engine with Python to automate AOI selection, NDVI computation, and forecasting, thereby enhancing accessibility and reproducibility.

# 2. Methods

## 2.1 Study Area and AOI Selection

An interactive map (geemap) allows users to draw a polygonal AOI. The AOI is retrieved as a GeoJSON geometry and used to filter Sentinel-2 collections in Earth Engine.

## 2.2 Data Acquisition and Preprocessing

Sentinel-2 Level-1C imagery is accessed using the collection `COPERNICUS/S2`. The pipeline filters by:
- AOI bounds.
- Date range (user-defined).
- Cloud cover threshold (`CLOUDY_PIXEL_PERCENTAGE`).

Multiple aggregation operators (median, mean, min, max, mosaic) are applied to generate a representative composite image.

## 2.3 Monthly NDVI Extraction

Monthly composites are created by filtering each calendar month and computing NDVI via:

\`\`\`python
ndvi = image.normalizedDifference(['B8', 'B4'])
\`\`\`

Mean NDVI values over the AOI are extracted using \`ee.Reducer.mean\`.

## 2.4 Forecasting with Prophet

Facebook Prophet fits the monthly NDVI time-series and produces a six-month forecast. The model handles seasonality and trend components automatically.

## 2.5 Multi-index Computation and Risk Detection

Additional indices are computed:
- NDBI (Built-up): \((B11 - B8) / (B11 + B8)\)
- NDWI (Water): \((B3 - B8) / (B3 + B8)\)
- NBR (Burn): \((B8 - B12) / (B8 + B12)\)
- SAVI (Soil-adjusted Vegetation): \((B8 - B4)/(B8 + B4 + 0.5) 	imes 1.5\)

Mean values over the AOI are flagged against thresholds:
| Index | Threshold      | Flag Condition         |
|-------|----------------|------------------------|
| NDVI  | < 0.4          | Vegetation stress      |
| NDBI  | > 0.3          | Urban expansion        |
| NDWI  | > 0            | Flood risk             |
| NBR   | < 0.1          | Burn indication        |
| SAVI  | < 0.3          | Low vegetation density |

# 3. Results

- **Interactive AOI selection** enabled by geemap.
- **Monthly NDVI series** plotted and tabulated.
- **Forecast results** illustrate expected vegetation dynamics.
- **Risk flags** identify areas of concern for vegetation stress, urbanization, flood potential, and fire damage.

# 4. Discussion

The integration of Earth Engine with Python automates NDVI analysis, reducing manual effort. Prophet's forecasting provides actionable insights for resource management. Thresholds can be customized per region. Limitations include reliance on optical imagery and cloud contamination.

# 5. Conclusion

This pipeline offers a reproducible, scalable solution for NDVI extraction and risk detection using open-source tools. Future work may integrate SAR data, refine risk thresholds, and extend to multi-sensor fusion.
