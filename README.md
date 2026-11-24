# ML vs DL for Chatbot Intent Classification: Controlled Comparison

Code for *Comparative Analysis of Machine Learning and Deep Learning Approaches for Intelligent Chatbot Development*. This project compares Machine Learning and Deep Learning approaches for chatbot intent classification using two benchmark datasets: Banking77 and CLINC150.

## Model Comparison

| Tier | Models | Script |
|---|---|---|
| Classical ML | TF-IDF + Multinomial NB / Logistic Regression / Linear SVM | `run_ml.py` |
| Task-trained neural | BiLSTM, TextCNN (GloVe 300d) | `run_nn.py` |
| Pretrained transformer | DistilBERT-base-uncased (fine-tuned) | `run_bert.py` |

## Protocol (Identical for Every Model)

- Official test sets are used once per final model. Banking77 has no official validation split, so 10% of the training data is held out (stratified, fixed seed). CLINC150 uses its official validation split.
- Training and validation rows whose normalized text also appears in the test set are removed using a leakage guard.
- Tuning: A maximum of 8 configurations per model are selected using validation accuracy at 100% training data and then reused at all data fractions. The settings are not re-tuned for low-data regimes.
- Learning curves: 5%, 10%, 25%, 50%, and 100% of training data; stratified and nested subsets; 5 seeds (42, 123, 456, 789, 1024). Classical models are deterministic, so 100% runs once.
- Metrics: Accuracy, macro-F1, training time, CPU inference latency (batch size 1, 1,000 utterances after 100 warm-up calls, p50/p95), and model size.

## Setup

```bash
pip install -r requirements.txt

# Download datasets
mkdir -p data && cd data

curl -L -o banking77_train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv

curl -L -o banking77_test.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

curl -L -o clinc150_full.json https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json

cd ..

# Download GloVe for run_nn.py
mkdir -p glove && cd glove

wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip

unzip glove.6B.zip glove.6B.300d.txt

cd ..
```

Check the licenses on the dataset pages before publishing.

## Run Order

```bash
# Smoke tests
python run_ml.py --quick && python run_nn.py --quick && python run_bert.py --quick

# Full experiments
python run_ml.py
python run_nn.py
python run_bert.py

# Generate analysis, tables, figures, and significance tests
python analysis.py
```

- `run_ml.py` runs classical Machine Learning models.
- `run_nn.py` runs the BiLSTM and TextCNN models.
- `run_bert.py` fine-tunes DistilBERT.
- `analysis.py` generates tables, figures, and significance tests in the `outputs/` folder.

All runners append results to `outputs/results_*.csv` and skip finished runs, allowing experiments to resume after a disconnect.

Delete `outputs/*_best_params.json` to force re-tuning.

Smoke tests (`--quick`) write to `outputs/quick_*.csv` and `outputs/quick_preds/`, not to the final result files.

If an older version put smoke-test rows into `results_*.csv`, or a laptop sleep caused an unusually high training time in a row, run:

```bash
python clean_results.py
```

This keeps a `.bak` copy. Removed runs can then be repeated, while completed runs are skipped.

## Experimental Results

The experiments have been completed using the Banking77 and CLINC150 datasets across classical Machine Learning, Neural Network, and pretrained transformer models.

The final experimental result files and best hyperparameter configurations are available in the `outputs/` folder.

### Output Files

| File | Description |
|---|---|
| `results_ml.csv` | Results of classical Machine Learning models |
| `results_nn.csv` | Results of BiLSTM and TextCNN models |
| `results_bert.csv` | Results of DistilBERT |
| `ml_best_params.json` | Best hyperparameters for Machine Learning models |
| `nn_best_params.json` | Best hyperparameters for Neural Network models |
| `bert_best_params.json` | Best hyperparameters for DistilBERT |

These files contain the recorded experimental results and selected model configurations used in the comparative study.

## Notes for the Paper

- Measure latency on one fixed CPU machine and report its specifications.
- `run_nn.py` and `run_bert.py` train on GPU, while `run_ml.py` trains on CPU. The `device` column records this information.
- If GloVe is missing, `run_nn.py` falls back to random embeddings and sets `glove=False` in the CSV. Do not report such runs as GloVe runs.
- Select the best model per family using validation accuracy (`analysis.py` does this), not test accuracy.

## Long Runs on a Laptop

- Keep the laptop plugged in.
- Set Sleep to "Never" while experiments are running.
- Pause OneDrive sync for the project folder.
- Keep the laptop lid open.

Training times are measured using wall-clock time, so laptop sleep can affect the recorded results.
