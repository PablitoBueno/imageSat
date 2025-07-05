# ─── IMPORTS ─────────────────────────────────────────────────────────────────
!pip install --quiet geemap earthengine-api ipywidgets prophet google-generativeai
import ee, geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from prophet import Prophet
from IPython.display import display, clear_output, Markdown
from PIL import Image
import google.generativeai as genai

# ─── GOOGLE EARTH ENGINE SETUP ──────────────────────────────────────────────
ee.Authenticate()
ee.Initialize(project='projeto-id')

# ─── GEMINI SETUP ───────────────────────────────────────────────────────────
genai.configure(api_key='GEMINI_API_KEY')  # 🔒 Substitua pela sua chave segura
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ─── MAPA INTERATIVO ────────────────────────────────────────────────────────
m = geemap.Map(center=[0,0], zoom=2)
m.add_basemap('SATELLITE')
m.add_draw_control()
display(m)

# ─── UI WIDGETS ─────────────────────────────────────────────────────────────
collection_input = widgets.Text(value='COPERNICUS/S2', description='Collection:')
start_date = widgets.Text(value='2022-01-01', description='Start Date:')
end_date = widgets.Text(value='2022-12-31', description='End Date:')
cloud_pct = widgets.IntSlider(value=20, min=0, max=100, description='Max Cloud %:')
operator = widgets.Dropdown(options=['median','mean','min','max','mosaic'], value='median', description='Operator:')
export_name = widgets.Text(value='sat_image', description='Export name:')
scale_m = widgets.IntSlider(value=10, min=10, max=100, step=10, description='Scale (m):')
run_button = widgets.Button(description='Run & Export', button_style='success')

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

# ─── FUNÇÕES DE PROCESSAMENTO ───────────────────────────────────────────────

def extract_monthly_ndvi(aoi, collection_id, start, end, scale):
    def monthly_mean(year, month):
        s = f'{year}-{month:02d}-01'
        e = f'{year}-{month:02d}-28'
        img = ee.ImageCollection(collection_id)\
            .filterBounds(aoi).filterDate(s, e).select('B8','B4')
        ndvi = img.map(lambda i: i.normalizedDifference(['B8','B4']).rename('nd'))\
                  .mean()\
                  .reduceRegion(ee.Reducer.mean(), aoi, scale)\
                  .get('nd')
        return [s, ndvi.getInfo()]
    months = pd.date_range(start=start, end=end, freq='MS')
    data = [monthly_mean(d.year, d.month) for d in months]
    df = pd.DataFrame(data, columns=['ds','y'])
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def forecast_ndvi(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=6, freq='MS')
    fc = m.predict(future)
    return m, fc

def compute_index(img, bands, name):
    return img.normalizedDifference(bands).rename(name)

def mean_over_aoi(img, aoi, scale):
    stat = img.reduceRegion(ee.Reducer.mean(), aoi, scale)
    return stat.values().get(0).getInfo()

def generate_gemini_insight(image_path, prompt):
    image = Image.open(image_path)
    response = gemini.generate_content([prompt, image])
    return response.text

# ─── CALLBACK PRINCIPAL ─────────────────────────────────────────────────────

def on_run(b):
    clear_output(wait=True)
    display(m, ui)

    aoi = m.user_roi
    if aoi is None:
        print("⚠️ Desenhe uma área primeiro.")
        return

    col = ee.ImageCollection(collection_input.value)\
        .filterBounds(aoi).filterDate(start_date.value, end_date.value)
    props = col.first().propertyNames().getInfo()
    if 'CLOUDY_PIXEL_PERCENTAGE' in props:
        col = col.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_pct.value))

    op = operator.value
    if op == 'median': img = col.median()
    elif op == 'mean': img = col.mean()
    elif op == 'min': img = col.min()
    elif op == 'max': img = col.max()
    else: img = col.mosaic()
    img = img.clip(aoi)

    # NDVI mensal + previsão
    df_ndvi = extract_monthly_ndvi(
        aoi, collection_input.value,
        start_date.value, end_date.value,
        scale_m.value
    )
    model_prophet, forecast = forecast_ndvi(df_ndvi)

    # ─ Visualização consolidada ─
    fig, axs = plt.subplots(1, 2, figsize=(12,4))

    # Gráfico NDVI + previsão
    model_prophet.plot(forecast, ax=axs[0])
    axs[0].set_title('Previsão NDVI')

    # Cálculo de índices
    ndvi = compute_index(img, ['B8','B4'], 'NDVI')
    ndbi = compute_index(img, ['B11','B8'], 'NDBI')
    ndwi = compute_index(img, ['B3','B8'], 'NDWI')
    nbr  = compute_index(img, ['B8','B12'], 'NBR')
    b8 = img.select('B8'); b4 = img.select('B4')
    savi = b8.subtract(b4).divide(b8.add(b4).add(0.5)).multiply(1.5).rename('SAVI')

    df_indices = pd.DataFrame({
        'Index': ['NDVI','NDBI','NDWI','NBR','SAVI'],
        'Mean': [
            mean_over_aoi(ndvi, aoi, scale_m.value),
            mean_over_aoi(ndbi, aoi, scale_m.value),
            mean_over_aoi(ndwi, aoi, scale_m.value),
            mean_over_aoi(nbr, aoi, scale_m.value),
            mean_over_aoi(savi, aoi, scale_m.value),
        ]
    })

    # Flags de risco
    df_indices['Risk_Flag'] = False
    df_indices.loc[df_indices.Index=='NDVI', 'Risk_Flag'] = df_indices.Mean < 0.4
    df_indices.loc[df_indices.Index=='NDBI', 'Risk_Flag'] = df_indices.Mean > 0.3
    df_indices.loc[df_indices.Index=='NDWI', 'Risk_Flag'] = df_indices.Mean > 0
    df_indices.loc[df_indices.Index=='NBR',  'Risk_Flag'] = df_indices.Mean < 0.1
    df_indices.loc[df_indices.Index=='SAVI', 'Risk_Flag'] = df_indices.Mean < 0.3

    # Gráfico dos índices
    axs[1].bar(df_indices['Index'], df_indices['Mean'], color=['green','gray','blue','red','olive'])
    axs[1].axhline(0, color='black', lw=0.5)
    axs[1].set_title("Índices Médios")

    # Salvar imagem
    fig.tight_layout()
    fig.savefig("grafico_final.png")
    display(fig)

    # Mostrar tabela
    print("📊 Tabela de índices com risco:")
    display(df_indices)

    # Insight automático com Gemini
    prompt = (
        "Este gráfico mostra a evolução temporal do NDVI (índice de vegetação) e os valores médios de índices ambientais "
        "como NDVI, NDBI, NDWI, NBR e SAVI em uma determinada área. Analise os padrões sazonais, a saúde da vegetação, "
        "sinais de urbanização ou degradação ambiental, e possíveis riscos identificados."
    )
    print("🧠 Gerando insight automático com Gemini...")
    insight = generate_gemini_insight("grafico_final.png", prompt)
    display(Markdown(f"### 🔎 Insight gerado automaticamente:\n\n{insight}"))

    # Exportar imagem para o Drive
    print("💾 Exportando imagem para o Google Drive…")
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
    print(f"🚀 Exportação iniciada: '{export_name.value}' em Drive/gee_exports")

run_button.on_click(on_run)
