import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from adjustText import adjust_text
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Fungsi untuk memproses gambar pada klasifikasi
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

# Fungsi untuk memuat data Clustering
def load_data():
    world_url = 'https://raw.githubusercontent.com/rifqi-qi/insight-clustering/refs/heads/main/world_map.geojson'
    clustered_data_url = 'https://raw.githubusercontent.com/rifqi-qi/insight-clustering/refs/heads/main/clustered_production_data.csv'
    world = gpd.read_file(world_url)
    clustered_df = pd.read_csv(clustered_data_url)
    return world, clustered_df

# Fungsi untuk membuat peta Clustering
def create_sea_map(world, clustered_df):
    sea_countries = ['Indonesia', 'Malaysia', 'Thailand', 'Vietnam', 'Philippines',
                     'Singapore', 'Brunei', 'Cambodia', 'Laos', 'Myanmar']
    sea_map = world[world['NAME'].isin(sea_countries)]
    sea_map = sea_map.merge(clustered_df[['Entity', 'Cluster', 'total_production', 'growth_rate']],
                            left_on='NAME', right_on='Entity', how='left')
    sea_map = sea_map.to_crs(epsg=3395)
    centroids = sea_map.geometry.centroid
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), constrained_layout=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sea_map.boundary.plot(ax=ax, linewidth=0, edgecolor="black")
    sea_map.plot(column='Cluster', ax=ax, legend=True, cmap='Spectral', 
                 edgecolor='darkgray', legend_kwds={'shrink': 0.8}, cax=cax)
    texts = []
    for centroid, label, total_prod, growth_rate, cluster in zip(
        centroids, sea_map['NAME'], sea_map['total_production'], 
        sea_map['growth_rate'], sea_map['Cluster']
    ):
        x, y = centroid.x, centroid.y
        annotation_text = (f"{label}\nCluster: {cluster}\n"
                           f"Total: {total_prod:.0f}\nGrowth: {growth_rate:.2f}%")
        texts.append(ax.text(x, y, annotation_text, fontsize=9, ha='center',
                              bbox=dict(facecolor='white', edgecolor='darkgray', 
                                        boxstyle="round,pad=0.3", alpha=0.8)))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    ax.scatter(centroids.x, centroids.y, color='green', s=50, label='Centroid')
    scalebar = ScaleBar(1, units="m", location='lower left', length_fraction=0.2)
    ax.add_artist(scalebar)
    fig.suptitle('Production Clustering Map of Southeast Asia', fontsize=15, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', alpha=1)
    return fig

# Fungsi Clustering
def clustering():
    st.title('Southeast Asia Production Clustering Map')
    try:
        world, clustered_df = load_data()
        fig = create_sea_map(world, clustered_df)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please check the GitHub URLs and ensure files are accessible")

# Model untuk klasifikasi
model1 = tf.keras.models.load_model('akbar.h5')
model2 = tf.keras.models.load_model('dana.h5')

# Navigasi utama
def main():
    st.sidebar.title("Navigasi")
    option = st.sidebar.radio("Pilih Halaman:", ["Clustering", "Klasifikasi 1", "Klasifikasi 2"])
    if option == "Clustering":
        clustering()
    elif option == "Klasifikasi 1":
        klasifikasi_1()
    elif option == "Klasifikasi 2":
        klasifikasi_2()

if __name__ == "__main__":
    main()
