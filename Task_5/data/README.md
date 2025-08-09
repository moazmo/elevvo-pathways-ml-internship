# Dataset Information

This project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

## Download Instructions

Due to the large size of the dataset (~300MB), it is not included in this repository. Please download it from Kaggle:

🔗 **[GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)**

## Directory Structure

After downloading, your `data/` directory should look like this:

```
data/
├── raw/
│   ├── Train/          # Training images organized by class folders (0-42)
│   ├── Test/           # Test images
│   ├── Meta/           # Metadata images
│   ├── Train.csv       # Training labels
│   ├── Test.csv        # Test labels
│   └── Meta.csv        # Class metadata
└── processed/          # Generated during preprocessing
    ├── dataset_stats.json
    ├── preprocessing_config.json
    └── split_indices.json
```

## Dataset Details

- **Classes**: 43 different German traffic signs
- **Training Images**: ~39,000 images
- **Test Images**: ~12,000 images
- **Image Size**: Variable (will be resized to 64x64 during preprocessing)
- **Format**: PPM files (converted to standard formats during loading)

## Setup Instructions

1. Download the dataset from the Kaggle link above
2. Extract the files to the `data/raw/` directory
3. Run the preprocessing notebooks in order:
   - `01_data_exploration.ipynb`
   - `02_preprocessing.ipynb`
   - `03_model_training.ipynb`
   - `04_model_comparison.ipynb`

The preprocessing will generate the required configuration files and trained models automatically.