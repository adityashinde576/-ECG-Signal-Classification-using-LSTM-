
# ECG Signal Classification and Visualization using LSTM

## Project Overview

This project focuses on **ECG (Electrocardiogram) signal classification** using a **Long Short-Term Memory (LSTM)** neural network. The model learns temporal patterns in ECG signals to classify different types of heartbeats.

The project also includes **ECG signal visualization**, predicted vs actual label comparison, and performance evaluation using accuracy and a confusion matrix.

This work is inspired by real-world healthcare applications such as:

* Cardiac monitoring systems
* Early arrhythmia detection
* Clinical decision support tools

---

## Problem Statement

ECG signals are sequential time-series data that represent the electrical activity of the heart. Traditional machine learning models struggle to capture long-term temporal dependencies in such data.

The goal of this project is to:

* Train an LSTM-based deep learning model
* Accurately classify ECG signals into multiple heartbeat categories
* Visualize ECG signals and model predictions for interpretability

---

## Dataset Information

Dataset used: **MIT-BIH Arrhythmia Dataset**

Source:

* Kaggle: [https://www.kaggle.com/datasets/shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

Files:

* `mitbih_train.csv`
* `mitbih_test.csv`

Dataset details:

* Each sample contains **187 time steps**
* Last column represents the class label
* Total classes: **5 heartbeat categories**
* Data is pre-segmented and labeled

---

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Project Architecture

1. Data Loading
2. Data Preprocessing

   * Feature scaling
   * Reshaping for LSTM input
3. LSTM Model Definition
4. Model Training
5. Model Evaluation
6. ECG Signal Visualization
7. Confusion Matrix Analysis

---

## LSTM Model Explanation

The LSTM network is chosen because:

* ECG signals are sequential time-series data
* LSTM can retain long-term dependencies
* It solves vanishing gradient problems of simple RNNs

Model structure:

* Input layer: ECG time steps
* LSTM layer: Learns temporal patterns
* Fully connected layer: Classification output
* Softmax via CrossEntropyLoss

---

## Installation and Setup Instructions

### Step 1: Clone the Repository

```
git clone https://github.com/your-username/ecg-lstm-classification.git
cd ecg-lstm-classification
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
```

### Step 3: Install Required Libraries

```
pip install torch numpy pandas matplotlib seaborn scikit-learn kaggle
```

---

## Dataset Setup Instructions

### Step 1: Download Dataset

1. Create a Kaggle account
2. Generate Kaggle API token
3. Place `kaggle.json` in:

```
C:\Users\<username>\.kaggle\
```

### Step 2: Download Dataset Using Command

```
kaggle datasets download -d shayanfazeli/heartbeat
```

### Step 3: Extract Files

Place the following files in a `data/` folder:

* `mitbih_train.csv`
* `mitbih_test.csv`

---

## How to Run the Project

### Step 1: Open Jupyter Notebook

```
jupyter notebook
```

### Step 2: Open the Notebook

```
Untitled31.ipynb
```

### Step 3: Run All Cells

* Kernel → Restart & Run All

The notebook will:

* Load and preprocess ECG data
* Train the LSTM model
* Evaluate performance
* Generate ECG visualizations
* Display confusion matrix

---

## Model Training

Training details:

* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Epochs: 20
* Batch Size: 128
* Device: GPU (if available) or CPU

During training, the loss decreases over epochs, indicating learning.

---

## Model Evaluation

Evaluation metrics:

* Test Accuracy
* Confusion Matrix

Example output:

```
Test Accuracy: 0.82+
```

Accuracy above **80%** is considered strong for ECG multi-class classification and suitable for academic and prototype-level clinical research.

---

## Visualization Output

The project generates:

* Raw ECG signal plots
* ECG signals with true labels
* ECG signals with predicted labels
* Confusion matrix heatmap

These visualizations help understand:

* ECG waveform structure
* Model prediction behavior
* Classification errors

---

## Real-World Applications

* Hospital ECG monitoring systems
* ICU early warning systems
* Arrhythmia detection tools
* Medical research and analysis
* AI-assisted cardiology tools

---

## Project Limitations

* Dataset is pre-segmented (not raw ECG stream)
* Not deployed as real-time application
* No chatbot interface implemented (visualization only)

---

## Future Enhancements

* Real-time ECG streaming support
* Attention-based LSTM or Transformer model
* Explainable AI using Grad-CAM or SHAP
* Web dashboard using Streamlit
* Chatbot-based ECG interpretation interface

---

## Author

Name: Aditya Shinde
Degree: B.Sc. Computer Science
Status: Final Year Computer Science Student
Domain: Machine Learning, Deep Learning, Healthcare AI

---

## License

This project is for educational and research purposes only.
Not intended for clinical diagnosis or medical decision-making.
