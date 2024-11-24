import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from branca.colormap import linear
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Fungsi untuk memuat data
def load_data():
    """Load data dari GitHub URLs"""
    world_url = 'https://raw.githubusercontent.com/rifqi-qi/insight-clustering/refs/heads/main/world_map.geojson'
    clustered_data_url = 'https://raw.githubusercontent.com/rifqi-qi/insight-clustering/refs/heads/main/clustered_production_data.csv'
    world = gpd.read_file(world_url)
    clustered_df = pd.read_csv(clustered_data_url)
    return world, clustered_df

# Fungsi untuk membuat peta interaktif
def create_interactive_map(world, clustered_df):
    """Buat peta interaktif dengan Folium berdasarkan clustering"""
    sea_countries = ['Indonesia', 'Malaysia', 'Thailand', 'Vietnam', 'Philippines',
                     'Singapore', 'Brunei', 'Cambodia', 'Laos', 'Myanmar']
    world['is_sea'] = world['NAME'].isin(sea_countries)
    sea_map = world.copy()

    sea_map = sea_map.merge(clustered_df[['Entity', 'Cluster', 'total_production', 'growth_rate', 'avg_annual_production']],
                            left_on='NAME', right_on='Entity', how='left')

    clusters = sea_map['Cluster'].dropna().unique()
    cluster_colormap = linear.Spectral_11.scale(min(clusters), max(clusters))
    cluster_colormap.caption = "Cluster Color Map"

    m = folium.Map(location=[5, 115], zoom_start=4, tiles="CartoDB positron", control_scale=True)
    Fullscreen().add_to(m)

    for _, row in sea_map.iterrows():
        color = (
            cluster_colormap(row['Cluster']) 
            if not pd.isna(row['Cluster']) 
            else "none"
        )
        tooltip_text = (
            f"<b>{row['NAME']}</b><br>"
            f"Cluster: {int(row['Cluster']) if not pd.isna(row['Cluster']) else 'N/A'}<br>"
            f"Total Production: {f'{int(row['total_production']):,}' if not pd.isna(row['total_production']) else 'N/A'}<br>"
            f"Avg Annual Production: {f'{int(row['avg_annual_production']):,}' if not pd.isna(row['avg_annual_production']) else 'N/A'}<br>"
            f"Growth Rate: {f'{row['growth_rate']:.2f}%' if not pd.isna(row['growth_rate']) else 'N/A'}<br>"
        )
        folium.GeoJson(
            data=row['geometry'].__geo_interface__,
            style_function=lambda feature, color=color: {
                "fillColor": color if color != "none" else "white",
                "color": "black",
                "weight": 0.5,
                "fillOpacity": 0.7 if color != "none" else 0.1,
            },
            tooltip=tooltip_text,
        ).add_to(m)

    m.add_child(cluster_colormap)
    return m

# Fungsi Clustering
def clustering():
    st.title('Southeast Asia Production Clustering Map')
    try:
        world, clustered_df = load_data()
        m = create_interactive_map(world, clustered_df)
        st_folium(m, width=1500, height=800)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please check the GitHub URLs and ensure files are accessible")

    
    st.title("Prediksi Konsumsi Ikan Tahunan untuk 5 Tahun ke Depan")
    data = {
        'Tahun': ['Tahun 1', 'Tahun 2', 'Tahun 3', 'Tahun 4', 'Tahun 5'],
        'Brunei': [0.499228, 0.491545, 0.452463, 0.354421, 0.165281],
        'Cambodia': [-2.810841, -5.846806, -9.410816, -11.892458, -16.040953],
        'Indonesia': [0.205084, -0.909621, -3.320006, -6.251882, -8.100619],
        'Laos': [-1.045137, -3.350134, -7.017705, -9.632481, -11.920973],
        'Malaysia': [-1.940506, -3.894156, -5.888247, -7.145387, -9.121558],
        'Myanmar': [-2.834271, -5.531009, -8.201625, -10.300562, -13.821525],
        'Philippines': [-0.088790, -0.277209, -0.548433, -0.874676, -1.047260],
        'Thailand': [0.028403, -0.498116, -1.572209, -2.788179, -3.786477],
        'Vietnam': [-0.694727, -2.266113, -4.983405, -7.333661, -9.227884],
    }
    df = pd.DataFrame(data)
    st.table(df)
    st.write("""
    Berdasarkan hasil prediksi konsumsi ikan untuk 5 tahun ke depan di beberapa negara, terlihat adanya tren yang bervariasi.
    - **Brunei**: Penurunan ringan.
    - **Cambodia**: Penurunan tajam.
    - **Indonesia**: Tren menurun stabil.
    - **Myanmar, Laos, Malaysia**: Penurunan terlihat cukup signifikan.
    """)

# Fungsi preprocessing gambar
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Fungsi Klasifikasi 1
def klasifikasi_1():
    st.title("🖼️ Klasifikasi 1: Spesies Ikan")
    st.markdown("""
    **Klasifikasi Gambar Spesies Ikan**
    - **Label 1:** Amphiprion clarkii
    - **Label 2:** Chaetodon lunulatus
    - **Label 3:** Chaetodon trifascialis
    """)
    uploaded_file = st.file_uploader("Upload gambar ikan (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.warning("⏳ Memproses gambar...")
        processed_image = preprocess_image(image)
        prediction = model1.predict(processed_image)
        class_index = np.argmax(prediction)
        labels = ["Label 1 (Amphiprion clarkii)", "Label 2 (Chaetodon lunulatus)", "Label 3 (Chaetodon trifascialis)"]
        predicted_label = labels[class_index]
        probability = prediction[0][class_index] * 100
        st.success(f"🎉 **Prediksi:** {predicted_label}")
        st.info(f"**Probabilitas:** {probability:.2f}%")

# Fungsi Klasifikasi 2
def klasifikasi_2():
    st.title("🖼️ Klasifikasi 2: Spesies Ikan")
    st.markdown("""
    **Klasifikasi Gambar Spesies Ikan**
    - **Label 1:** Chromis Chrysura
    - **Label 2:** Dascyllus Reticulatus
    - **Label 3:** Plectroglyphidodon Dickii
    """)
    uploaded_file = st.file_uploader("Upload gambar ikan (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.warning("⏳ Memproses gambar...")
        processed_image = preprocess_image(image)
        prediction = model2.predict(processed_image)
        class_index = np.argmax(prediction)
        labels = ["Label 1 (Chromis Chrysura)", "Label 2 (Dascyllus Reticulatus)", "Label 3 (Plectroglyphidodon Dickii)"]
        predicted_label = labels[class_index]
        probability = prediction[0][class_index] * 100
        st.success(f"🎉 **Prediksi:** {predicted_label}")
        st.info(f"**Probabilitas:** {probability:.2f}%")

# Sidebar
with st.sidebar:
    st.title("Navigasi")
    menu = st.radio("Pilih Fitur:", ["Clustering dan Prediksi", "Klasifikasi 1", "Klasifikasi 2"])

# Menjalankan fitur berdasarkan pilihan pengguna
if menu == "Clustering dan Prediksi":
    clustering()
elif menu == "Klasifikasi 1":
    klasifikasi_1()
elif menu == "Klasifikasi 2":
    klasifikasi_2()
