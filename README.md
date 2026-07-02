# AI Construction Site Feasibility Prediction System

## Overview
This system predicts the construction feasibility score for any GPS coordinate in India. It evaluates live climate, seismic, soil, and ecological data to determine site viability. The core machine learning architecture relies on a stacking ensemble, leveraging Random Forest, XGBoost, and Extra Trees as base models, with a Ridge regression meta-learner. The final model achieved 91.23% R² accuracy with a ±1.99 point average error, providing data-driven foundation recommendations and visual risk scenarios for early-stage planning.

## Features
- **Site Evaluation**: Analyzes soil quality, climate risks, seismic activity, and ecological factors.
- **Feasibility Score**: Outputs a comprehensive score out of 100 indicating site suitability.
- **Risk Assessment**: Generates risk profiles for flooding, earthquakes, cyclones, and wildlife conflicts.
- **Foundation Recommendations**: Provides data-driven advice on the appropriate foundation type based on soil bearing capacity.
- **PDF Reports**: Generates automated feasibility reports.
- **Voice Reports**: Provides an automated Tamil voice script summarizing the findings.

## Tech Stack
- **Backend**: Python, Flask, SQLAlchemy
- **Machine Learning**: Scikit-Learn, XGBoost, PyTorch (Soil Imaging)
- **APIs**: NASA POWER, Bhuvan, USGS Earthquake Hazards Program

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables by copying `.env.example` to `.env` and adding your API keys.

### Running the Application
```bash
python app.py
```
The application will be available at `http://127.0.0.1:5000/`.
