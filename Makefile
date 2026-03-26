.PHONY: install data features train evaluate api dashboard clean all

install:
	pip install -r requirements.txt

data:
	python -m src.data.generate_data

features:
	python -m src.features.feature_engineering

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

clean:
	rm -rf data/raw/*.csv data/processed/*.csv data/processed/*.parquet
	rm -rf models/*.joblib models/*.pkl
	rm -rf results/figures/*.png results/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: data features train evaluate
