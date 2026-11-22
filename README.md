# Ensemble Learning Dashboard - Sri Lankan Used-Vehicle Prices

## Source

The real dataset used by this dashboard is
`data/car_price_dataset.csv`, sourced from the
[SriLankan Vehicle (Car) Price Dataset](https://www.kaggle.com/datasets/prasadnirmal/srilankan-second-vehiclecar-price-dataset)
on Kaggle. The project uses listings from ikman.lk and Riyasewana.com.

## Run (local)
1. python -m venv venv && source venv/bin/activate      (Windows: venv\Scripts\activate)
2. pip install -r requirements.txt
3. Put your REAL data in data/vehicles_sri_lanka.csv
   columns: make, model, year, mileage, engine, fuel, transmission, district, price
   (no data yet? `python generate_sample_data.py` makes a synthetic TEST file - never submit its results)
   The Kaggle Sri Lankan vehicle dataset is supported directly from `data/car_price_dataset.csv`.
4. Notebook: jupyter notebook Ensemble_Dashboard.ipynb  -> Kernel > Restart & Run All
   (or: python train_models.py). This creates models/ and results/.
5. Dashboard: streamlit run app.py

## Google Colab
Upload all files, run `!pip install -r requirements.txt`, then open the notebook and Run All.

## Files
train_models.py / Ensemble_Dashboard.ipynb  full pipeline (clean, EDA, tune, bagging, boosting, stacking, hybrids, evaluation, save)
hybrid.py        RF -> ANN hybrid model class
app.py           Streamlit dashboard
scraper/scrape_template.py  polite scraper template (untested, adjust selectors)
timeseries_rf_arima.py      alternative path for rainfall / temperature
make_notebook.py            rebuilds the .ipynb from train_models.py

AI disclosure: code skeleton generated with Claude; data, running, results and interpretation by the student.
