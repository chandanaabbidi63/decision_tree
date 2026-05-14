# Decision Tree Regression (Streamlit)

This repo contains a Streamlit app that performs **Decision Tree Regression** using the pre-trained model in `model.pkl`.

## Files
- `app.py` - Streamlit UI (regression only)
- `model.pkl` - trained `DecisionTreeRegressor` model
- `requirements.txt` - Python dependencies

## Run locally
1. Create & activate a virtual environment (recommended):
   - Windows (PowerShell):
     - `python -m venv venv`
     - `venv\Scripts\Activate.ps1`

2. Install dependencies:
   - `pip install -r requirements.txt`

3. Start Streamlit:
   - `streamlit run app.py --server.port 8501 --server.address 127.0.0.1`

4. Open the URL shown in the terminal (typically `http://127.0.0.1:8501`).

## Deploy to Streamlit Community Cloud
- Ensure the following are in the project root:
  - `app.py`
  - `model.pkl`
  - `requirements.txt`

- Cloud deployment automatically runs `streamlit run app.py`.

## Notes
- The app expects **10 numeric inputs**: `Age, Sex, BMI, Blood Pressure, S1..S6`.
- If `model.pkl` is missing, the app will show a clear error message.

