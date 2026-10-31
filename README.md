# FaceSense AI

FaceSense AI is an end-to-end computer vision system for:

- Face detection and recognition of registered people
- Gender classification and approximate age estimation
- Emotion detection and liveness checks
- Attendance logging
- Dashboard and API integrations

## Project structure

```text
FaceSenseAI/
├── app/
│   ├── api/          # API routes and request handlers
│   ├── core/         # Application configuration and shared core logic
│   ├── models/       # Model loading and inference interfaces
│   ├── services/     # Face analysis and attendance services
│   └── utils/        # Shared utility functions
├── data/
│   ├── raw/          # Original input data
│   ├── processed/    # Preprocessed data
│   └── known_faces/  # Registered face images
├── models/           # Downloaded or trained model files
├── notebooks/        # Exploration and training notebooks
├── tests/             # Automated tests
├── ui/                # Dashboard and user interface
├── config.py          # Paths, image settings, thresholds, and model names
└── requirements.txt   # Python dependencies
```

## Getting started

1. Install Python 3.11, then create and activate a virtual environment in
   **Command Prompt**:

   ```bash
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   In **PowerShell**, activate it with:

   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. Upgrade pip and install the pinned dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Add registered face images to `data/known_faces/`.

4. Place downloaded or trained model artifacts in `models/`.

5. Start building the API, services, and dashboard in their respective folders.

## Preparing UTKFace data

Place UTKFace images named like `age_gender_race_date.jpg` in `data/raw/`,
then run:

```bash
python data/prepare_dataset.py
```

The script skips invalid or corrupt images, resizes valid images to 128x128,
creates stratified 70/15/15 train, validation, and test splits under
`data/processed/`, and writes `data/processed/labels.csv`. Use
`--input-dir`, `--output-dir`, or `--random-state` to customize the defaults.

## Configuration

Update [config.py](config.py) to change image dimensions, confidence thresholds,
attendance deduplication behavior, or model identifiers. Paths are resolved
relative to the project root.
