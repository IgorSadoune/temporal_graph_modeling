# Abstract

This paper presents a self-contained approach to temporal graph modeling tailored for sparse maritime port networks, with a specific application to the Great Lakes–St. Lawrence (GLSL) corridor. Despite the critical role of these inland waterway systems, decision-making is often hampered by data sparsity and intermittent traffic patterns that standard predictive models fail to capture. Leveraging Automatic Identification System (AIS) data spanning 12 years, we propose a methodology that addresses these challenges through spatial node aggregation and a novel temporal k-core decomposition algorithm. We conduct a comparative analysis of three deep learning architectures: a baseline Long Short-Term Memory (LSTM) network, a sequential Graph Attention Network integrated with a Gated Recurrent Unit (GAT-GRU), and a parallel Graph WaveNet model. Our empirical results indicate that the GAT-GRU architecture offers superior predictive performance in this sparse setting, outperforming both the LSTM baseline and Graph WaveNet. Beyond forecasting, we demonstrate how this framework enables counterfactual analysis to simulate network responses to exogenous shocks, providing a robust foundation for strategic planning and resilience assessment in maritime supply chains.

# Instructions

## Repo Structure

```
.
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── main.sh                            # Main script to run full experiments
├── main_sample.sh                     # Script to run experiments with sample data
├── paper_website_version.tex          # LaTeX source for paper
├── paper_website_version.pdf          # Compiled paper PDF
│
├── data/                              # Data directory
│   ├── ais_o_d.csv                   # Raw AIS origin-destination data (CSV)
│   ├── ais_o_d.parquet               # Raw AIS origin-destination data (Parquet)
│   ├── processed_ais_o_d.csv         # Preprocessed AIS data (CSV)
│   ├── processed_ais_o_d.parquet     # Preprocessed AIS data (Parquet)
│   └── graph_sequence.pt             # Temporal graph sequence (PyTorch)
│
├── src/                               # Source code directory
│   ├── config.yaml                   # Configuration file for full experiments
│   ├── config_sample.yaml            # Configuration file for sample data experiments
│   │
│   ├── preprocessing.py              # Data preprocessing script
│   ├── get_temporal_sequence.py      # Temporal graph sequence generation
│   │
│   ├── train_lstm.py                 # LSTM model training script
│   ├── train_gatgru.py               # GAT-GRU model training script
│   ├── train_wavenet.py              # Graph WaveNet model training script
│   │
│   ├── inference.py                  # Model inference script
│   ├── counterfactual_gatgru.py      # Counterfactual analysis script
│   ├── plots.py                      # Visualization and plotting utilities
│   │
│   └── modules/                      # Model architecture modules
│       ├── lstm.py                   # LSTM architecture implementation
│       ├── gatgru.py                 # GAT-GRU architecture implementation
│       ├── wavenet.py                # Graph WaveNet architecture implementation
│       └── temporal_graphs.py        # Temporal graph utilities
│
├── outputs/                           # Model outputs and results
│   ├── lstm_best_model.pt            # Best LSTM model checkpoint
│   ├── lstm_metrics.json             # LSTM evaluation metrics
│   ├── gatgru_best_model.pt          # Best GAT-GRU model checkpoint
│   ├── gatgru_metrics.json           # GAT-GRU evaluation metrics
│   ├── wavenet_best_model.pt         # Best WaveNet model checkpoint
│   ├── wavenet_metrics.json          # WaveNet evaluation metrics
│   ├── inference_results.json        # Inference results
│   ├── inference_result_figure.png   # Inference visualization
│   └── losses_over_epoch.png         # Training loss curves
│
├── computers_environment_urban_systems/  # Journal submission materials
│   ├── manuscript.tex                # Main manuscript LaTeX source
│   ├── manuscript.pdf                # Compiled manuscript PDF
│   ├── cover_letter.tex              # Cover letter LaTeX source
│   ├── cover_letter.pdf              # Compiled cover letter PDF
│   ├── highlights.tex                # Research highlights LaTeX source
│   ├── highlights.pdf                # Compiled highlights PDF
│   ├── title_page.tex                # Title page LaTeX source
│   ├── title_page.pdf                # Compiled title page PDF
│   ├── library.bib                   # Bibliography file
│   ├── arxiv.sty                     # ArXiv style file
│   ├── utphys.bst                    # Bibliography style file
│   ├── inference_result_figure.png   # Figure for manuscript
│   └── losses_over_epoch.png         # Figure for manuscript
│
└── venv_temporal_graph/               # Python virtual environment (excluded from version control)
``` 


## Virtual Environment Setup
The virtual environment is already setup on the local server @tc.cirano.qc.ca, so you just need to activate it before attempting to run any scripts or experiments.

### Activating the Virtual Environment
To activate the virtual environment, go to the root of the directory and use the following command:

```bash
source venv_temporal_graph/bin/activate
```

### Deactivating the Virtual Environment
To deactivate the virtual environment, simply use the following command:

```bash
deactivate
```

### Setting up the Environment
If for some reason you need to set up again or elsewehre, use:

```bash
python3 -m venv venv_temporal_graph
source venv_temporal_graph/bin/activate
pip install -r requirements.txt
```

## Running the Experiments

### Full Experiments
You can run the experiments in the paper by executing

```bash
bash main.sh
```

### Using sample data
In order to test the code and to run the experiments on sample data, you can use

```bash
bash main_sample.sh
```

> [!CAUTION]
> Running `main_sample.sh` will overwrite existing output files with results from the sample data, which will replace the results shown in the paper. If you want to preserve the original results, make a copy of your output files before running the sample data script. You will need to run the full experiment with the complete dataset again to retrieve the paper's results.

## Running Idividual Scripts

### Using full dataset
To run an individual script, go to the root of the directory and use for instance

```bash
python3 src/train_lstm.py
```
in order to train the LSTM model. Look at the other scripts to run specific tasks or experiments.

### Using sample dataset
Use

```bash
python3 src/train_lstm.py --config src/config_sample.yaml
```

> [!CAUTION]
> Running with `config_sample.yaml` will overwrite existing output files with results from the sample data, which will replace the results shown in the paper. If you want to preserve the original results, make a copy of your output files before running the sample data script. You will need to run the full experiment with the complete dataset again to retrieve the paper's results.

