# Personality Prediction

Logistic regression classifier that predicts a personality type from structured behavioral data.

The pipeline encodes categorical fields, scales numeric fields, ranks features with ANOVA (`f_classif`), keeps the strongest features, and trains a logistic regression model. The dataset in `data/personality.csv` is generated with a fixed random seed so the training run is reproducible. It is a portfolio dataset for this project, not a clinical personality instrument.

## Run

```bash
pip install -r requirements.txt
python generate_data.py
python train.py
python predict.py --interest Books --social-events 1 --hours-alone 10 --friends 2 --posts 0 --outdoor-days 0 --stage-fear Yes --drained Yes
```

`train.py` writes `models/personality_model.joblib` and `models/metrics.json`. On this dataset the held-out accuracy is about 75 percent. That number is measured by `train.py`; it is not copied from another run.

## Test

```bash
python -m unittest discover -s tests
```
