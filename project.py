import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import ee
import pandas as pd
from datetime import datetime
from PIL import Image
from sklearn.linear_model import LinearRegression
from skimage import exposure
from esda.moran import Moran
from libpysal.weights import lat2W, W

# Step 1: Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize()

# Step 2: Load Satellite Image with Multiple Dates
def fetch_multitemporal_data(region_coords, start_date, end_date):
    region = ee.Geometry.Point(region_coords)
    collection = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    # Calculate NDWI and NDVI for each image
    def calculate_indices(image):
        green = image.select('B3')
        blue = image.select('B2')
        nir = image.select('B8')
        red = image.select('B4')
        ndwi = green.subtract(blue).divide(green.add(blue)).rename('NDWI')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return image.addBands([ndwi, ndvi])

    collection = collection.map(calculate_indices)
    return collection

# Example region and dates
region_coords = [32.52549, 40.49416]
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
            scale=10
        ).get('NDWI').getInfo()
        if mean_ndwi:
            dates.append(date)
            ndwi_values.append(mean_ndwi)
    
    return dates, ndwi_values

# Get NDWI time series data
dates, ndwi_values = get_time_series_data(collection)

# Trend Analysis using Linear Regression
dates_numeric = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
regressor = LinearRegression().fit(dates_numeric, ndwi_values)
trend = regressor.predict(dates_numeric)

# Time Series Visualization
plt.figure(figsize=(10, 6))
plt.plot(dates, ndwi_values, label="NDWI")
plt.plot(dates, trend, label="Trend (Linear Regression)", linestyle="--")
plt.xlabel("Date")
plt.ylabel("NDWI")
plt.legend()
plt.title("NDWI Time Series with Trend")
plt.show()

# Step 3: Local Image Classification with Gradient Boosting
local_image_path = 'C:/Users/harun.can/Desktop/Sentinel_TIFF.tif'

def load_image(path):
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile
    return image, profile

image, profile = load_image(local_image_path)

# Extract bands for classification
red = image[0, :, :]
green = image[1, :, :]
blue = image[2, :, :]
ndwi = (green - blue) / (green + blue + 1e-5)
ndwi_filtered = gaussian_filter(ndwi, sigma=1)

# Generate synthetic labels
labels = np.random.choice([0, 1], size=ndwi_filtered.size)
features = np.stack([ndwi_filtered.ravel(), red.ravel(), green.ravel(), blue.ravel()], axis=1)

# Train Gradient Boosting Classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbc.fit(X_train, y_train)

# Evaluate model
y_pred = gbc.predict(X_test)
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict the full map
classified_map = gbc.predict(features).reshape(ndwi_filtered.shape)

# Step 4: Visualize the classified map with Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(classified_map, cmap='viridis')
plt.colorbar()
plt.title("Gradient Boosting Classified Map")
plt.show()

# Normalize the classified map
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
    crs=profile['crs'],
    transform=profile['transform'],
) as dst:
    dst.write(normalized_map, 1)

print(f"Classified map saved at {output_path}")

# Spatial weights using a raster-like structure
rows, cols = classified_map.shape
w = lat2W(rows, cols)

# Flatten the raster for Moran's I calculation
flattened_map = classified_map.ravel()

# Remove missing data (if any)
valid_data = flattened_map > 0  # Assuming non-zero values are valid
filtered_map = flattened_map[valid_data]

# Rebuild the spatial weights for valid data
filtered_weights_sparse = w.sparse[valid_data, :][:, valid_data]
filtered_weights = W.from_sparse(filtered_weights_sparse)  # Convert to a libpysal.weights.W object

# Compute Moran's I
if filtered_map.size == filtered_weights.n:  # Ensure dimensions match
    moran = Moran(filtered_map, filtered_weights)
    print("Moran's I:", moran.I)
else:
    print(f"Dimension mismatch: filtered_map size = {filtered_map.size}, filtered_weights size = {filtered_weights.n}")

# Step 7: Change Detection
image1 = collection.filterDate('2023-01-01', '2023-06-30').median()
image2 = collection.filterDate('2023-07-01', '2023-12-31').median()
change_map = image1.select('NDWI').subtract(image2.select('NDWI'))

# Optional: Step 8: Web-based Dashboard
try:
    import streamlit as st

    st.title("Deforestation Analysis Dashboard")
    st.write("Visualize NDWI Trends and Classification Results")

    # NDWI Trend Chart
    st.subheader("NDWI Time Series")
    st.line_chart(pd.DataFrame({'Date': dates, 'NDWI': ndwi_values}).set_index('Date'))

    # Display the classified map
    st.subheader("Classified Map")
    classified_map_image = Image.open(output_path)
    st.image(classified_map_image, caption="Gradient Boosting Classified Map")

except ImportError:
    print("Streamlit is not installed. Install it to use the web interface: pip install streamlit")
