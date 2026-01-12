# AM-BiLSTM-MC for Lower Limb Motion Recognition

This repository contains the implementation of the **AM-BiLSTM-MC** (Attention Mechanism - Bidirectional LSTM - Markov Chain) model proposed in the paper:

> **[Action-Specific Attention and Logic-Corrected Long-Sequence Lower Limb Action Recognition for Daily-Life Rehabilitation]** >

## 1. Introduction
This project focuses on recognizing human lower limb motion intent using a hybrid deep learning approach. The model processes data from **6-channel sEMG** and **3-channel IMU** sensors to accurately classify continuous motion sequences.

**Key Features:**
* **Data Source:** Multi-modal sensor fusion (sEMG + IMU).
* **Model:** Hybrid AM-BiLSTM-MC architecture.
* **Goal:** Continuous motion intent recognition and logical sequence correction.

## 2. Prerequisites & Environment

### Python (Deep Learning Model)
The deep learning model is implemented in Python (PyTorch/TensorFlow).
* Python 3.8+
* PyTorch / TensorFlow 1.11+
* NumPy, Pandas, Scikit-learn

To install dependencies, run:
```bash
pip install -r requirements.txt

### MATLAB Environment (Signal Processing)
MATLAB R2022b (or newer recommended)
Signal Processing Toolbox

3. Project Structure
├── data/               # Place sample data here (Do NOT upload full patient data)
├── preprocessing/      # MATLAB scripts for filtering and segmentation
├── model/              # Python scripts for AM-BiLSTM construction
├── training/           # Training loops and validation scripts
├── results/            # Confusion matrices and plots
├── main.py             # Main entry point for training/testing
└── README.md           # This file

4. Usage
Step 1: Data Preparation
Run the MATLAB script preprocessing/data_process.m to filter and segment the raw signals. (Note: Due to privacy regulations, only a sample dataset is provided in this repository.)

Step 2: Training the Model
To train the AM-BiLSTM model, run:
python main.py --mode train

Step 3: Evaluation
To evaluate the model and apply Markov Chain correction:
python main.py --mode test

5. License
This project is licensed under the MIT License - see the LICENSE file for details.
