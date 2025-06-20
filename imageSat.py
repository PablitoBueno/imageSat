# 0️⃣ Install dependencies
!pip install --quiet geemap earthengine-api ipywidgets prophet

# 1️⃣ Imports & init
import ee, geemap
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from prophet import Prophet

# Authenticate & initialize with your Project ID
ee.Authenticate()
ee.Initialize(project='project-id')

# ───────────────────────────────────────────────────
# 2️⃣ Interactive AOI drawing
m = geemap.Map(center=[0,0], zoom=2)
m.add_basemap('SATELLITE')
m.add_draw_control()
display(m)

# ───────────────────────────────────────────────────
# 3️⃣ UI widgets (all keyword args)

collection_input = widgets.Text(
    value='COPERNICUS/S2',
    description='Collection:'
)

start_date = widgets.Text(
    value='2022-01-01',
    description='Start Date:'
)

end_date = widgets.Text(
    value='2022-12-31',
    description='End Date:'
)

cloud_pct = widgets.IntSlider(
    value=20,
    min=0,
    max=100,
    description='Max Cloud %:'
)

operator = widgets.Dropdown(
    options=['median','mean','min','max','mosaic'],
    value='median',
    description='Operator:'
)

export_name = widgets.Text(
    value='sat_image',
    description='Export name:'
)

scale_m = widgets.IntSlider(
    value=10,
    min=10,
    max=100,
    step=10,
    description='Scale (m):'
)

run_button = widgets.Button(
    description='Run & Export',
    button_style='success'
)

ui = widgets.VBox([
    collection_input,
    widgets.HBox([start_date, end_date]),
    cloud_pct,
    operator,
    export_name,
    scale_m,
    run_button
])
display(ui)

# ───────────────────────────────────────────────────
# 4️⃣ Monthly NDVI extraction + forecast

def extract_monthly_ndvi(aoi, collection_id, start, end, scale):
    """Return DataFrame with monthly mean NDVI for AOI."""
    def monthly_mean(year, month):
        s = f'{year}-{month:02d}-01'
        e = f'{year}-{month:02d}-28'
        img = ee.ImageCollection(collection_id)\
                .filterBounds(aoi)\
                .filterDate(s, e)\
                .select('B8','B4')  # Sentinel-2: NIR=B8, Red=B4
        ndvi = img.map(lambda i: i.normalizedDifference(['B8','B4']))\
                  .mean()\
                  .reduceRegion(
                      ee.Reducer.mean(), aoi, scale
                  )\
                  .get('nd')
        return [s, ndvi.getInfo()]
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    months = pd.date_range(start_dt, end_dt, freq='MS')
    data = [monthly_mean(d.year, d.month) for d in months]
    df = pd.DataFrame(data, columns=['ds','y'])
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def forecast_ndvi(df):
    """Fit Prophet and forecast next 6 months."""
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=6, freq='MS')
    fc = m.predict(future)
    return m, fc

# ───────────────────────────────────────────────────
# 5️⃣ Main pipeline callback

def on_run(b):
    clear_output(wait=True)
    display(m, ui)

    # AOI
    aoi = m.user_roi
    if aoi is None:
        print("⚠️ Draw your AOI first.")
        return

    # Image collection
    col = ee.ImageCollection(collection_input.value)\
            .filterBounds(aoi)\
            .filterDate(start_date.value, end_date.value)
    # Cloud filter
    props = col.first().propertyNames().getInfo()
    if 'CLOUDY_PIXEL_PERCENTAGE' in props:
        col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_pct.value))

    # Apply operator & clip
    op = operator.value
    if op=='median': img = col.median()
    elif op=='mean': img = col.mean()
    elif op=='min':   img = col.min()
    elif op=='max':   img = col.max()
    else:            img = col.mosaic()
    img = img.clip(aoi)

    # Display on map
    m.clear_layers(); m.add_basemap('SATELLITE')
    m.addLayer(aoi, {'color':'red'}, 'AOI')
    vis = {'bands':['B4','B3','B2'],'min':0,'max':3000}
    m.addLayer(img, vis, 'Processed')
    display(m)

    # Extract & forecast NDVI
    print("📈 Extracting monthly NDVI…")
    df = extract_monthly_ndvi(
        aoi, collection_input.value,
        start_date.value, end_date.value,
        scale_m.value
    )
    print(df.tail())
    m_prophet, fc = forecast_ndvi(df)
    fig = m_prophet.plot(fc)
    display(fig)

    # Export GeoTIFF
    print("💾 Exporting image…")
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=export_name.value,
        folder='gee_exports',
        fileNamePrefix=export_name.value,
        region=aoi.bounds().getInfo()['coordinates'],
        scale=scale_m.value,
        crs='EPSG:4326'
    )
    task.start()
    print(f"🚀 Export started as '{export_name.value}' in Drive/gee_exports")

run_button.on_click(on_run)

# ───────────────────────────────────────────────────
# End of notebook

# ───────────────────────────────────────────────────
# Block 2: Index Measurement & Risk Detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Ensure AOI is drawn
aoi = m.user_roi
if aoi is None:
    raise Exception("⚠️ Draw your AOI on the map first!")

# 2️⃣ Select a single representative image (median) from previously filtered collection
# (reuse collection from Block 1 or redefine here)
col = ee.ImageCollection(collection_input.value) \
        .filterBounds(aoi) \
        .filterDate(start_date.value, end_date.value)
# cloud filter if available
props = col.first().propertyNames().getInfo()
if 'CLOUDY_PIXEL_PERCENTAGE' in props:
    col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_pct.value))
img = col.median().clip(aoi)

# 3️⃣ Functions to compute indices
def compute_index(img, bands, name):
    """Normalized difference index: (b1 - b2)/(b1 + b2)"""
    nd = img.normalizedDifference(bands).rename(name)
    return nd

# Compute each index
ndvi = compute_index(img, ['B8','B4'], 'NDVI')    # Sentinel-2 NIR/Red
ndbi = compute_index(img, ['B11','B8'], 'NDBI')   # SWIR/NIR
ndwi = compute_index(img, ['B3','B8'], 'NDWI')    # Green/NIR
nbr  = compute_index(img, ['B8','B12'], 'NBR')    # NIR/SWIR2
# SAVI: (NIR - Red)/(NIR + Red + L)*(1+L), L=0.5
b8 = img.select('B8').rename('NIR')
b4 = img.select('B4').rename('RED')
savi = b8.subtract(b4).divide(b8.add(b4).add(0.5)).multiply(1.5).rename('SAVI')

# 4️⃣ Reduce each index to mean over AOI
def mean_over_aoi(index_img):
    stat = index_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale_m.value
    )
    return stat.values().get(0).getInfo()

results = {
    'Index': ['NDVI','NDBI','NDWI','NBR','SAVI'],
    'Mean': [
        mean_over_aoi(ndvi),
        mean_over_aoi(ndbi),
        mean_over_aoi(ndwi),
        mean_over_aoi(nbr),
        mean_over_aoi(savi)
    ]
}

df = pd.DataFrame(results)

# 5️⃣ Simple risk thresholds (tweak as needed)
df['Risk_Flag'] = False
# Vegetation stress if NDVI < 0.4
df.loc[df.Index=='NDVI','Risk_Flag'] = df.Mean < 0.4
# Urban expansion concern if NDBI > 0.3
df.loc[df.Index=='NDBI','Risk_Flag'] = df.Mean > 0.3
# Flood risk if NDWI > 0
df.loc[df.Index=='NDWI','Risk_Flag'] = df.Mean > 0
# Fire burn indication if NBR < 0.1
df.loc[df.Index=='NBR','Risk_Flag'] = df.Mean < 0.1
# Low vegetation with SAVI < 0.3
df.loc[df.Index=='SAVI','Risk_Flag'] = df.Mean < 0.3

# 6️⃣ Display table
print("📊 Mean index values and risk flags for your AOI:")
display(df)

# 7️⃣ Bar chart
plt.figure(figsize=(8,4))
plt.bar(df['Index'], df['Mean'], color=['green','grey','blue','red','olive'])
plt.axhline(0, color='black', linewidth=0.5)
plt.title('Mean Indices over AOI')
plt.ylabel('Mean value')
plt.show()
