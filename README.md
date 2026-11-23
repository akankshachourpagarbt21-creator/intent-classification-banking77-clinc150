# ML vs DL for Chatbot Intent Classification: controlled comparison

Code for *Comparative Analysis of Machine Learning and Deep Learning Approaches for Intelligent
Chatbot Development*. Two benchmarks (Banking77, CLINC150 in-scope), three model tiers, one protocol.

| Tier | Models | Script |
|---|---|---|
| Classical ML | TF-IDF + Multinomial NB / Logistic Regression / Linear SVM | `run_ml.py` |
| Task-trained neural | BiLSTM, TextCNN (GloVe 300d) | `run_nn.py` |
| Pretrained transformer | DistilBERT-base-uncased (fine-tuned) | `run_bert.py` |

## Protocol (identical for every model)
- Official test sets, used once per final model. Banking77 has no official validation split, so 10% of
  train is held out (stratified, fixed seed). CLINC150 uses its official validation split.
- Train/val rows whose normalized text also appears in the test set are dropped (leakage guard).
- Tuning: <= 8 configurations per model, chosen on validation accuracy at 100% data, then reused at all
  fractions. (Limitation: the settings are not re-tuned for low-data regimes.)
- Learning curves: 5/10/25/50/100% of train; stratified and nested subsets; 5 seeds
  (42, 123, 456, 789, 1024). Classical models are deterministic, so 100% runs once.
- Metrics: accuracy, macro-F1, training time, CPU latency (batch 1, 1,000 utterances after 100 warm-up
  calls, raw text to label, p50/p95), model size.

## Setup
```bash
pip install -r requirements.txt
# data (already in ./data if you cloned the repo with it, otherwise):
mkdir -p data && cd data
curl -L -o banking77_train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
curl -L -o banking77_test.csv  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
curl -L -o clinc150_full.json  https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
cd ..
# GloVe for run_nn.py
mkdir -p glove && cd glove && wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip && unzip glove.6B.zip glove.6B.300d.txt && cd ..
```
Check the licenses on the dataset pages before publishing.

## Run order
```bash
python run_ml.py --quick && python run_nn.py --quick && python run_bert.py --quick   # smoke tests
python run_ml.py            # minutes on CPU
python run_nn.py            # use a GPU (Colab T4 is fine)
python run_bert.py          # use a GPU
python analysis.py          # tables + figures + significance tests -> outputs/
```
All runners append to `outputs/results_*.csv` and skip finished runs, so you can resume after a disconnect.
Delete `outputs/*_best_params.json` to force re-tuning.

Smoke tests (`--quick`) write to `outputs/quick_*.csv` and `outputs/quick_preds/`, never to the real
results. If an older version put smoke-test rows into `results_*.csv`, or a laptop sleep left an absurd
training time in a row, run `python clean_results.py` (it keeps a `.bak` copy), then re-run the same
commands: removed runs are redone and finished ones are skipped.

Long runs on a laptop: keep it plugged in, set Sleep to "Never", pause OneDrive sync for the project
folder, and keep the lid open. Training times are wall-clock, so a sleep ruins them.

## Notes for the paper
- Measure latency on one fixed CPU machine and report its specification. Note that `run_nn.py` and
  `run_bert.py` train on GPU while `run_ml.py` trains on CPU; the `device` column records this.
- If GloVe is missing, `run_nn.py` falls back to random embeddings and sets `glove=False` in the CSV.
  Do not report such runs as GloVe runs.
- Select the "best model per family" by validation accuracy (analysis.py does this), not by test accuracy.
