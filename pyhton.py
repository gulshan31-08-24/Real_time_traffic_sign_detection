import os
import pandas as pd

# === CONFIGURATION ===
image_dir = "myData"              # Folder containing 43 class-wise subfolders
label_csv_path = "labels.csv"     # CSV mapping class_id to label name
output_csv_path = "image_summary.csv"  # Output file for Power BI

# === LOAD LABELS ===
label_df = pd.read_csv(label_csv_path)
label_df.columns = ["class_id", "label_name"]  # Rename for consistency

# === SCAN IMAGE FOLDERS ===
records = []

for class_id in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_id)
    if not os.path.isdir(class_path):
        continue
    try:
        class_id_int = int(class_id)
    except ValueError:
        print(f"Skipping folder {class_id}: not a valid integer class")
        continue

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            records.append({
                "image_path": os.path.abspath(img_path),
                "class_id": class_id_int
            })

# === CREATE DATAFRAME AND MERGE LABELS ===
df = pd.DataFrame(records)
df = df.merge(label_df, on="class_id", how="left")

# === SAVE TO CSV ===
df.to_csv(output_csv_path, index=False)

print(f"✅ Saved summary of {len(df)} images to '{output_csv_path}'")
