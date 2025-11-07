import splitfolders

# Path to the original dataset folder containing class-wise subfolders
input_folder = r"./data/raw"

# Output folder structure base (train, val, test folders will be created inside)
output_folder = r"./data"

# Split ratio: 70% train, 15% val, 15% test
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.7, 0.15, 0.15))

print("Data successfully split into train, val, and test inside the data/ folder!")
