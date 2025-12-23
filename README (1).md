# 🩺 Comprehensive TB Chest X-ray Classification

### Biomedical Machine Learning Project

This project implements a **comprehensive Tuberculosis (TB) detection
system** using **machine learning** on **chest X-ray images**.\
It is designed to support **biomedical analysis and screening** by
combining image preprocessing, machine learning classification, clinical
performance evaluation, and biomedical interpretation.

------------------------------------------------------------------------

## 📌 Project Overview

Tuberculosis remains a major global health challenge. Early and accurate
detection using chest X-ray imaging is crucial. This project applies a
**Random Forest classifier** to processed chest X-ray images to
distinguish between:

-   **Normal lungs**
-   **Tuberculosis-infected lungs**

The system evaluates results using **clinically meaningful metrics** and
provides **interactive prediction** for new images.

------------------------------------------------------------------------

## ✨ Key Features

-   Chest X-ray image classification (Normal vs Tuberculosis)
-   Image preprocessing (resizing, normalization, flattening)
-   Random Forest machine learning model
-   Comprehensive clinical performance metrics
-   Biomedical interpretation of diagnostic errors
-   Medical visualizations and plots
-   Interactive testing of user-provided X-ray images
-   Model and result persistence for future use

------------------------------------------------------------------------

## 📊 Dataset

**Dataset Used:** TB Chest Radiography Database

### Required Folder Structure

    TB_Chest_Radiography_Database/
    │
    ├── Normal/
    │   ├── image1.png
    │   ├── image2.jpg
    │
    ├── Tuberculosis/
    │   ├── image1.png
    │   ├── image2.jpg

-   Images must be grayscale chest X-rays
-   Supported formats: `.png`, `.jpg`, `.jpeg`

------------------------------------------------------------------------

## ⚙️ Requirements

Install the required Python libraries:

``` bash
pip install numpy matplotlib seaborn scikit-learn opencv-python joblib
```

**Python Version:** 3.7 or higher

------------------------------------------------------------------------

## ▶️ How to Run the Project

1.  Clone the repository:

``` bash
git clone https://github.com/yourusername/BiomedicalML_TB_ChestXray.git
cd BiomedicalML_TB_ChestXray
```

2.  Update the dataset path in the code:

``` python
dataset_path = r"PATH_TO/TB_Chest_Radiography_Database"
```

3.  Run the script:

``` bash
python tb_chest_xray_classification.py
```

------------------------------------------------------------------------

## 🔍 Code Execution Flow

1.  Load chest X-ray images and assign labels
2.  Preprocess images (resize to 64×64, normalize, flatten)
3.  Split dataset into training and testing sets
4.  Train Random Forest classifier
5.  Evaluate clinical performance metrics
6.  Generate medical and statistical visualizations
7.  Save trained model and performance results
8.  Enable interactive prediction for new X-ray images

------------------------------------------------------------------------

## 📈 Evaluation Metrics

The system evaluates performance using:

-   Accuracy
-   Sensitivity (Recall)
-   Specificity
-   Precision
-   F1-score
-   ROC-AUC
-   Precision-Recall AUC

These metrics are interpreted in a **clinical context** to assess
screening suitability.

------------------------------------------------------------------------

## 🧑‍⚕️ Biomedical Interpretation

The program analyzes: - False negatives (missed TB cases) - False
positives (healthy cases flagged as TB) - Clinical risks and population
impact - Deployment recommendations (screening, triage, or improvement
needed)

This ensures results are meaningful in real-world healthcare scenarios.

------------------------------------------------------------------------

## 🧪 Interactive Prediction

After training, users can test their own chest X-ray images:

1.  Place an X-ray image in the same folder as the script
2.  Run the program
3.  Enter the image filename when prompted
4.  Receive:
    -   Diagnosis (Normal / Tuberculosis)
    -   Confidence probabilities
    -   Visual prediction output

⚠️ This feature is for **educational and research purposes only**.

------------------------------------------------------------------------

## 📁 Output Files

After execution, the following files are generated:

-   `tb_chest_xray_model.pkl` -- Trained model
-   `clinical_performance_results.pkl` -- Performance metrics
-   `medical_tb_analysis_comprehensive.png` -- Evaluation plots
-   `user_prediction_result.png` -- User prediction visualization

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is intended **only for academic and research purposes**. It
is **not approved for clinical diagnosis**.\
Always consult qualified healthcare professionals for medical decisions.

------------------------------------------------------------------------

## 👤 Author

-   **Name:** 1, Amanuel Asfachew
              2, Abeb bizan
-   **Field:** Biomedical Engineering / Machine Learning
-   **GitHub:** https://github.com/AI Assignment
