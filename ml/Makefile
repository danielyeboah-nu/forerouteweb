PY ?= python3
PIP ?= pip3
PORT_SERVE ?= 5001
PORT_UI ?= 5000
TRACKING_URI ?= sqlite:///$(CURDIR)/mlflow.db
MODEL_NAME ?= ForeRoute-SegmentRisk
BOSTON_MODEL_NAME ?= ForeRoute-BostonRisk
BOSTON_DATA ?= $(CURDIR)/data/boston_crash_dataset.csv
COMBINED_MODEL_NAME ?= ForeRoute-Combined
COMBINED_DATA ?= $(CURDIR)/data/combined_segments.csv

.PHONY: install train boston-data train-boston combine-data train-combined calibrate-thresholds explain evidently aadt mlflow-ui serve serve-boston serve-combined smoke clean

install:
	$(PIP) install -r requirements.txt

train:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) train.py

boston-data:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) build_boston_dataset.py

train-boston:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) train.py \
		--data $(BOSTON_DATA) \
		--label-col label \
		--registry-name $(BOSTON_MODEL_NAME) \
		--experiment foreroute-boston-risk

combine-data:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) combine_datasets.py

train-combined: combine-data
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) train.py \
		--data $(COMBINED_DATA) \
		--label-col label \
		--registry-name $(COMBINED_MODEL_NAME) \
		--experiment foreroute-combined

calibrate-thresholds:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) calibrate_thresholds.py

explain:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) explainability_analysis.py

evidently:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) $(PY) regenerate_evidently_reports.py

aadt:
	$(PY) build_massdot_aadt.py

mlflow-ui:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow ui --backend-store-uri $(TRACKING_URI) --port $(PORT_UI)

serve:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow models serve \
		-m "models:/$(MODEL_NAME)@production" \
		-p $(PORT_SERVE) \
		--env-manager local \
		--host 0.0.0.0

serve-boston:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow models serve \
		-m "models:/$(BOSTON_MODEL_NAME)@production" \
		-p $(PORT_SERVE) \
		--env-manager local \
		--host 0.0.0.0

serve-combined:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow models serve \
		-m "models:/$(COMBINED_MODEL_NAME)@production" \
		-p $(PORT_SERVE) \
		--env-manager local \
		--host 0.0.0.0

smoke:
	curl -s -X POST http://localhost:$(PORT_SERVE)/invocations \
		-H "Content-Type: application/json" \
		-d '{"dataframe_split": {"columns": ["temperature","precipitation_type","precipitation_intensity","wind_speed","visibility","humidity","dew_point","road_type","segment_distance_m"], "data": [[-3.0,"snow",4.2,28.0,0.6,90.0,-4.0,"bridge",5000.0]]}}'
	@echo

clean:
	rm -rf mlruns mlartifacts mlflow.db __pycache__
