import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import ee
import pandas as pd
from datetime import datetime
from PIL import Image

# Step 1: Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize()

# Step 2: Load Satellite Image with Multiple Dates
def fetch_multitemporal_data(region_coords, start_date, end_date):
    region = ee.Geometry.Point(region_coords)  # Using Point for specific coordinates
    collection = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Limit cloud cover

    def calculate_ndwi(image):
        green = image.select('B3')  # Green band (B3)
        blue = image.select('B2')  # Blue band (B2)
        ndwi = green.subtract(blue).divide(green.add(blue)).rename('NDWI')
        return image.addBands(ndwi)

    collection = collection.map(calculate_ndwi)
    return collection

# Example region and dates
region_coords = [32.52549, 40.49416]  # Coordinates adjusted per your request
start_date = '2023-01-01'
end_date = '2023-12-31'

collection = fetch_multitemporal_data(region_coords, start_date, end_date)

# Visualize NDWI Time Series
def get_time_series_data(collection):
    images = collection.toList(collection.size())
    dates = []
    ndwi_values = []
    for i in range(collection.size().getInfo()):
        img = ee.Image(images.get(i))
        date = datetime.strptime(img.date().format('YYYY-MM-dd').getInfo(), '%Y-%m-%d')
        mean_ndwi = img.select('NDWI').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee.Geometry.Point(region_coords),
            scale=9.55  # Adjusting the scale as requested (9.55 m/px)
        ).get('NDWI').getInfo()
        if mean_ndwi:
            dates.append(date)
            ndwi_values.append(mean_ndwi)
    
    return dates, ndwi_values

# Get NDWI time series data
dates, ndwi_values = get_time_series_data(collection)

# Step 3: Local Image Classification with Gradient Boosting
local_image_path = 'C:/Users/harun.can/Desktop/Sentinel_TIFF.tif'  # Local image file for classification

def load_image(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

image, profile = load_image(local_image_path)

# Extract bands for classification (Make sure these bands correspond to the correct indices)
red = image[0, :, :]  # Red band (Usually B4)
green = image[1, :, :]  # Green band (Usually B3)
blue = image[2, :, :]   # Blue band (Usually B2)

# Calculate NDWI (using Green and Blue bands)
ndwi = (green - blue) / (green + blue + 1e-5)  # NDWI index based on Green and Blue

# Smoothing
ndwi_filtered = gaussian_filter(ndwi, sigma=1)

# Generate synthetic labels (example)
labels = np.random.choice([0, 1], size=ndwi_filtered.size)
features = np.stack([ndwi_filtered.ravel()], axis=1)

# Train Gradient Boosting Classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)

# Evaluate model
y_pred = gbc.predict(X_test)
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred))

# Predict the full map
classified_map = gbc.predict(features).reshape(ndwi_filtered.shape)

# Step 4: Visualize the classified map with Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(classified_map, cmap='viridis')  # Use 'viridis' colormap for better contrast
plt.colorbar()
plt.title("Gradient Boosting Classified Map")
plt.show()

# Normalize the classified map for better visibility
from skimage import exposure
normalized_map = exposure.rescale_intensity(classified_map, 
                                            in_range=(classified_map.min(), classified_map.max()), 
                                            out_range=(0, 255))
normalized_map = normalized_map.astype(np.uint8)

# Step 5: Save the classified map as GeoTIFF
output_path = 'classified_map_gradient_boosting.tif'
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=normalized_map.shape[0],
    width=normalized_map.shape[1],
    count=1,
    dtype='uint8',
    crs=profile['crs'],  # Coordinate system from the original image
    transform=profile['transform'],  # Geo-transform from the original image
) as dst:
    dst.write(normalized_map, 1)

print(f"Classified map saved at {output_path}")

# Optional: Step 6: Web-based Dashboard for Visualization (if Streamlit is installed)
try:
    import streamlit as st

    st.title("Deforestation Analysis Dashboard")
    st.write("Visualize NDWI Trends and Classification Results")

    # NDWI Trend Chart
    st.subheader("NDWI Time Series")
    st.line_chart(pd.DataFrame({'Date': dates, 'NDWI': ndwi_values}).set_index('Date'))

    # Display the classified map with proper visualization
    st.subheader("Classified Map")

    # Open the classified map to display in Streamlit
    classified_map_image = Image.open(output_path)
    st.image(classified_map_image, caption="Gradient Boosting Classified Map")

except ImportError:
    print("Streamlit is not installed. Install it to use the web interface: pip install streamlit")
