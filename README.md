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

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add registered face images to `data/known_faces/`.

4. Place downloaded or trained model artifacts in `models/`.

5. Start building the API, services, and dashboard in their respective folders.

## Configuration

Update [config.py](config.py) to change image dimensions, confidence thresholds,
attendance deduplication behavior, or model identifiers. Paths are resolved
relative to the project root.

