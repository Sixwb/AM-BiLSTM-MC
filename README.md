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

### MATLAB Environment (Signal Processing)
* MATLAB R2022b (or newer recommended)
* Signal Processing Toolbox

## 3. Project Structure
├── data/
├── preprocessing/
├── inference/
├── train/
├── LICENSE.txt
└── README.md

## 4. License
This project is licensed under the MIT License - see the LICENSE file for details.

## 5. Note
As subsequent research will build upon the current work, only partial content is available for this project at this time. Full extended codebases may be released upon the completion of future studies.
