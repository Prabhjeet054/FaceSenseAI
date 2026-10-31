"""Check that the project's Python dependencies can be imported."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


LIBRARIES = {
    "opencv-python": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "tensorflow": "tensorflow",
    "scikit-learn": "sklearn",
    "insightface": "insightface",
    "onnxruntime": "onnxruntime",
    "mediapipe": "mediapipe",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "streamlit": "streamlit",
    "sqlalchemy": "sqlalchemy",
    "pytest": "pytest",
    "python-multipart": "multipart",
}


def check_library(distribution_name: str, import_name: str) -> bool:
    """Import one library and print its status and installed version."""
    try:
        module = import_module(import_name)
        try:
            installed_version = version(distribution_name)
        except PackageNotFoundError:
            installed_version = getattr(module, "__version__", "unknown")
        print(f"PASS {distribution_name} ({installed_version})")
        return True
    except Exception as error:
        print(f"FAIL {distribution_name}: {error}")
        return False


def main() -> int:
    results = [
        check_library(distribution_name, import_name)
        for distribution_name, import_name in LIBRARIES.items()
    ]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} libraries passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
