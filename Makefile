PY ?= python

install:            ## Instala dependencias de desarrollo
	pip install -r requirements.txt

first-run:          ## PRIMERA VEZ: descarga datos reales, construye dataset y entrena el primer champion
	$(PY) -m aqi.download --all
	$(PY) -m aqi.ingest --reset --all
	$(PY) -m aqi.train --force-promote --all

ingest:             ## Ingesta incremental (+ validación pandera)
	$(PY) -m aqi.ingest --all

train:              ## Entrena challenger y compara con el champion (umbral normal +0.005)
	$(PY) -m aqi.train --all

demo-reject:        ## Demo: mismos datos, config y seed que el champion -> challenger idéntico -> RECHAZADO
	$(PY) -m aqi.train

demo-promote:       ## Demo: umbral de demostración -1.0 -> PROMOVIDO (ver docs/DEMO_SCRIPT.md)
	$(PY) -m aqi.train --n-trials 8 --seed 7 --min-delta -1.0

test:
	pytest --cov=aqi

lint:
	ruff check .

serve:              ## API + UI en http://localhost:7860
	uvicorn main:app --host 0.0.0.0 --port 7860 --reload

mlflow-ui:
	mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

docker:
	docker build -t aqi-app . && docker run --rm -p 7860:7860 aqi-app
