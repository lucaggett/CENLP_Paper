# ZuCo-BERT Correlation Analysis


This repository contains a Python script to analyze correlations between human eye-tracking data from the ZuCo dataset and attention layers of BERT models. The analysis includes both monolingual (`bert-base-uncased`) and multilingual (`bert-base-multilingual-cased`) BERT models.

## Table of Contents
- [Overview](#overview)
- [File Structure](#file-structure)
- [Setup](#setup)
- [Usage](#usage)

---

## Overview
The script performs the following tasks:
1. Validates the file structure of the `results_zuco` folder.
2. Loads eye-tracking data for each participant using the `utils_ZuCo` library.
3. Uses BERT models to calculate attention weights at the word level.
4. Computes Spearman correlations between human attention data and BERT attention weights.
5. Outputs the results to a CSV file and generates visualizations.

---

## File Structure
The following folder structure is required for the script to run correctly:
```plaintext
.
├── results_zuco/
│   ├── task2/
│   │   ├── file1.mat
│   │   ├── file2.mat
│   │   └── ... (total 12 .mat files)
│   └── task_materials/
│       └── relations_labels_task2.csv
├── requirements.txt
└── zuco_bert_analysis.py
```
---

## Setup

### Prerequisites
- Python 3.10 or higher
- Pip package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/lucaggett/CENLP_Paper.git
   cd zuco-bert-analysis
   ```
2. Install the required Packages:
  ```bash
   pip install -r requirements.txt
  ```

## Usage

### Running the Script
1. Execute the script:
   ```bash
   python zuco_bert_analysis.py
   ```
After providing the required paths, the script should run without further interaction.
