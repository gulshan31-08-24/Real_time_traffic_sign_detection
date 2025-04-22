from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, IntegerType
import os
import numpy as np
import cv2

######################## CONFIGURATION ########################

image_dir = "myData"          # Path to dataset with class-wise folders
image_size = (32, 32)         # Target image dimensions
output_X_path = "X_data.npy"  # File to save preprocessed images
output_y_path = "y_data.npy"  # File to save labels

#################### INIT SPARK SESSION #######################

spark = SparkSession.builder.appName("TrafficSignPreprocessing").getOrCreate()

#################### COLLECT IMAGE PATHS ######################

image_records = []
for class_id in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_id)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        full_path = os.path.join(class_path, img_file)
        image_records.append(Row(path=full_path, label=int(class_id)))

image_df = spark.createDataFrame(image_records)

#################### DEFINE PREPROCESSING UDF ##################

def preprocess_image(path):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255.0
        return img.flatten().tolist()
    except:
        return np.zeros(image_size[0] * image_size[1]).tolist()

preprocess_udf = udf(preprocess_image, ArrayType(FloatType()))

# Apply the preprocessing function to each image path
processed_df = image_df.withColumn("image", preprocess_udf("path"))

##################### COLLECT AND SAVE #########################

# Collect the results into driver memory
results = processed_df.select("image", "label").collect()

# Convert to NumPy arrays
X = np.array([np.array(row["image"]).reshape(image_size[0], image_size[1], 1) for row in results])
y = np.array([row["label"] for row in results])

# Save preprocessed data for later use
np.save(output_X_path, X)
np.save(output_y_path, y)

spark.stop()
