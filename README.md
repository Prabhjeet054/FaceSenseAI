# FaceSense AI

FaceSense AI is an end-to-end computer vision system for:

- Face detection and recognition of registered people
- Gender classification and approximate age estimation
- Seven-class facial emotion classification
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

1. Install Python 3.11, then create the virtual environment with that
   interpreter. Do not reuse a `.venv` created with Python 3.14; TensorFlow
   2.16.1 does not provide a compatible wheel for Python 3.14.

   On **macOS/Linux**:

   ```bash
   deactivate 2>/dev/null || true
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

   In **Windows Command Prompt**:

   ```bash
   deactivate
   rmdir /s /q .venv
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

   In **PowerShell**, activate it with:

   ```bash
   deactivate
   Remove-Item -Recurse -Force .venv
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Confirm that the new environment is using Python 3.11, then install the
   pinned dependencies:

   ```bash
   python --version
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   The version check must print `Python 3.11.x`. If it prints `Python 3.14`,
   deactivate the environment, remove `.venv`, and repeat step 1.

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

## Training the gender model

After preparing the dataset, train the CNN with:

```bash
python3 train_gender.py
```

The script applies horizontal flips, small rotations, and brightness
augmentation to the training split only. It uses early stopping and saves the
best checkpoint to `models/gender.h5`; accuracy and loss curves are saved to
`models/gender_training_curves.png`. Training options such as `--epochs`,
`--batch-size`, `--patience`, and `--seed` can be overridden from the command
line.

## Training the age model

Train the regression head first, then fine-tune the top 30 MobileNetV2 layers:

```bash
python3 train_age.py
```

Validation MAE is printed after every epoch. A `ReduceLROnPlateau` scheduler
is used in both phases, and the best checkpoint is saved to
`models/age.h5`. Override `--head-epochs`, `--fine-tune-epochs`,
`--batch-size`, or `--fine-tune-learning-rate` as needed.

## Evaluating trained models

After both models and the processed test split are available, run:

```bash
python3 evaluate.py
```

The evaluator writes `reports/metrics.json`, `reports/gender_confusion_matrix.png`,
and `reports/age_error_by_bucket.png`. Model and dataset paths can be
overridden with `--gender-model`, `--age-model`, and `--data-dir`.

## Processing a video

Process every frame of a video and save annotated output plus detection
metadata:

```bash
python3 process_video.py input.mp4 \
  --output-video reports/annotated_video.mp4 \
  --output-csv reports/detections.csv
```

The CSV contains `timestamp`, `name`, `gender`, and `age` for every detection.

## Running the API

Set both privacy-related environment variables before starting the API. The
API fails closed if either key is missing. Generate a Fernet-compatible
encryption key with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Then configure the values in the shell that starts Uvicorn:

```bash
export FACESENSE_API_KEY="replace-with-a-long-random-api-key"
export FACESENSE_ENCRYPTION_KEY="replace-with-the-generated-fernet-key"
```

On Windows PowerShell, use `$env:FACESENSE_API_KEY = "..."` and
`$env:FACESENSE_ENCRYPTION_KEY = "..."` instead. Keep the encryption key
stable; changing it makes previously stored embeddings unavailable.

After installing the pinned dependencies and placing the trained model files
in `models/`, start the FastAPI server with:

```bash
python3 -m uvicorn app.api.main:app --reload
```

In another terminal, start the Streamlit dashboard:

```bash
python3 -m streamlit run ui/app.py
```

Set `FACESENSE_API_URL` to point the dashboard at a different backend, or
change the URL in the dashboard sidebar. The dashboard has Upload and Analyze,
Live Camera, Register Person, and Logs and Stats pages.

The API provides `GET /health`, `POST /analyze`, `POST /register`,
`POST /attendance`, `GET /people`, `DELETE /people/{person_id}`, `GET /logs`,
and `GET /stats`. Registration and attendance require a live frame sequence;
photo and screen spoof attempts are rejected.
Recognition events are stored in SQLite when `/analyze` identifies faces.
`/logs` accepts optional `date_from`, `date_to`, and `person_id` filters;
`/stats` returns counts grouped by gender, age group, and day. Image uploads must be JPEG,
PNG, or WebP files no larger than 10 MB. Registration accepts three to five
photos as multipart form data using the `name`, `person_id`, and `photos`
fields.

Daily attendance is available at `GET /attendance?attendance_date=YYYY-MM-DD`
and can be downloaded as an Excel workbook from
`GET /attendance/export?attendance_date=YYYY-MM-DD`. A person is marked once
per day on their first successful live recognition; repeats within the configured
cooldown are ignored.

Registration requires an explicit consent checkbox in the UI and stores the
consent timestamp with the registered person. All API routes require the
`X-API-Key` header. Recognition logs are automatically deleted after
`config.LOG_RETENTION_DAYS` days. Use
`DELETE /people/{person_id}/data` to delete a person's registration,
recognition logs, and attendance records.

## Inference optimization and benchmarking

The pipeline batches ArcFace embeddings and gender/age predictions for all
faces in an image. The optimized path also keeps a bounded in-memory cache of
embeddings for repeated face crops. Use the baseline/optimized benchmark with
a directory of sample images:

```bash
python benchmark_pipeline.py data/benchmark --repetitions 3
```

The script reports average milliseconds per image and per detected face before
and after batching and caching. Convert the trained gender and age Keras
models to optimized TensorFlow Lite artifacts with:

```bash
python convert_models.py --gender-model models/gender.h5 \
  --age-model models/age.h5 --output-dir models/tflite
```

The generated `.tflite` files can be loaded through
`app.core.model_runtime.TFLiteModel` when deploying a lightweight inference
runtime.

## Configuration

Update [config.py](config.py) to change image dimensions, confidence thresholds,
attendance deduplication behavior, or model identifiers. Paths are resolved
relative to the project root.
