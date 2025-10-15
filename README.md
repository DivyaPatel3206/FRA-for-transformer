# FRA-for-transformer

This project provides a FastAPI-based web service for **predicting transformer fault types** using synthetic or real Frequency Response Analysis (FRA) data.

## Features

* Supports multiple fault types:

  * Normal
  * Axial Displacement
  * Radial Deformation
  * Core Grounding
  * Turn-to-Turn Fault
* Returns **fault type** and **probability (0-100%)**.
* Compatible with CSV datasets containing FRA magnitude and phase data.

## Installation

1. Clone the repository:

```bash
git clone <repo_url>
cd <repo_folder>
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install fastapi uvicorn pandas numpy torch
```

## Model Preparation

* Place your PyTorch model file (`fra_model.pt`) in the project folder.
* Ensure your model was trained with features in the order: Magnitude (dB), Phase (°).
* Confirm the order of fault classes matches the `fault_labels` array in the code.

## Usage

1. Run the API server:

```bash
uvicorn main:app --reload
```

2. Send a POST request to `/predict` with your CSV file:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@fra_dataset.csv"
```

3. Example JSON response:

```json
{
  "status": "success",
  "filename": "fra_dataset.csv",
  "result": {
    "fault_type": "Core Grounding",
    "probability": 87.45
  }
}
```

## CSV Format

Your FRA CSV should have columns:

```
Frequency (Hz), Magnitude (dB), Phase (°)
10, -0.5, 3.1
20, -0.6, 2.8
...
```

## Notes

* The model expects **preprocessed features** as flattened magnitude + phase arrays.
* Probability output is **rounded to 2 decimals**.
* Synthetic datasets can be used for testing.

## License

MIT License
