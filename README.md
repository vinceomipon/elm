# elm

## Directory Structure

```text
aleasat-elm-analysis/
│
├── data/                         # Dedicated Image Repository
│   ├── 1_raw/                    # Unprocessed ELM photos straight from the camera 
│   │
│   ├── 2_processed/              # Intermediate steps for debugging
│   │   ├── hsv_channels/         # Extracted Hue, Saturation, Value mono images 
│   │   └── masks/                # Binary (0 and 1) black & white cluster masks
│   │
│   └── 3_output/                 # Final product for reporting
│       └── annotations/          # Images with the "ELM Coverage: XX.XX%" overlay 
│
├── src/
│   ├── __init__.py
│   ├── main.py              # Main entry point to run the pipeline
│   │
│   ├── core/                # Core mathematical algorithms
│   │   ├── __init__.py
│   │   └── kmeans.py        # Custom or wrapped K-means algorithm implementation
│   │
│   ├── processing/          # Image manipulation modules
│   │   ├── __init__.py
│   │   ├── image_loader.py  # Reads, resizes, and saves images
│   │   ├── color_space.py   # Handles RGB to HSV conversions
│   │   └── segmenter.py     # High-level segmentation pipeline (panel vs. illuminated)
│   │
│   └── metrics/             # Calculations and output generation
│       ├── __init__.py
│       └── area_calculator.py # Computes pixel ratios and expected power metrics
│
├── tests/                   # Unit tests for core processing steps
│   ├── test_kmeans.py
│   └── test_segmenter.py
│
├── requirements.txt         # Dependencies (OpenCV, NumPy, PyYAML, etc.)
└── README.md                # Project documentation
```