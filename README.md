# Hybrid Machine Learning Model for In-Hospital Mortality Prediction

## 🎯 Project Overview

This project develops a **hybrid predictive model** combining empirical medical risk calculation with machine learning to predict in-hospital mortality. Using the MIMIC-IV dataset (545,497 hospitalizations), we achieved **98.46% AUROC** while maintaining clinical interpretability.

### Key Results
- **AUROC:** 0.9846
- **Sensitivity:** 94.9% (catches 95% of deaths)
- **Specificity:** 93.7%
- **NPV:** 99.9% (when predicting survival, correct 999/1000 times)

### Why This Matters
- ✅ **Early identification** of high-risk patients at hospital admission
- ✅ **Interpretable predictions** - clinicians can see WHY someone is high-risk
- ✅ **Outperforms traditional scores** - 10% improvement over APACHE-style scoring
- ✅ **Fair across demographics** - consistent performance across age groups

---

## 📁 Project Structure
```
mortality-prediction/
├── notebooks/          # Jupyter notebooks (analysis pipeline)
├── figures/           # Visualizations and charts
├── models/            # Trained ML models (.pkl files)
├── results/           # Summary reports and metrics
├── docs/              # Documentation for collaborators
├── .gitignore         # Files NOT to upload
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

---

## 📊 Data Source

This project uses **MIMIC-IV** (Medical Information Mart for Intensive Care IV), a publicly available critical care database from Beth Israel Deaconess Medical Center.

### ⚠️ Data Access Required

**The data files are NOT included in this repository** due to PhysioNet data use agreements.

To reproduce this analysis:

1. **Complete CITI Training:**
   - Go to: https://physionet.org/about/citi-course/
   - Complete "Data or Specimens Only Research" course

2. **Request MIMIC-IV Access:**
   - Create PhysioNet account: https://physionet.org/register/
   - Request access: https://physionet.org/content/mimiciv/2.2/
   - Approval typically takes 1-3 days

3. **Download Required Files:**
```
   mimic-iv-2.2/hosp/
   ├── admissions.csv.gz
   ├── patients.csv.gz
   ├── diagnoses_icd.csv.gz
   ├── drgcodes.csv.gz
   └── d_icd_diagnoses.csv.gz
```

4. **Place in Project Root:**
   - Extract `.gz` files or leave compressed (code handles both)
   - Place in the same folder as the notebooks

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Lab or Jupyter Notebook
- MIMIC-IV data access (see above)

### Installation

1. **Clone this repository:**
```bash
   git clone https://github.com/[YOUR_USERNAME]/mortality-prediction.git
   cd mortality-prediction
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Add MIMIC-IV data files** (see Data Access section)

4. **Run notebooks in order:**
```bash
   jupyter lab
```
   - Open `notebooks/01_exploratory_analysis.ipynb`
   - Run all cells (Shift + Enter)
   - Proceed through notebooks 02-05

---

## 📓 Notebook Pipeline

### **Notebook 01: Exploratory Data Analysis**
- Loads MIMIC-IV data files
- Performs quality checks
- Computes descriptive statistics
- Visualizes patient demographics, diagnoses, outcomes

**Key Output:** Clean merged dataset with 545,497 admissions

---

### **Notebook 02: Empirical Risk Calculation**
- Calculates diagnosis-specific mortality rates (Variable X, Y)
- Computes DRG severity mortality rates (Variable Z)
- Generates baseline risk score per hospitalization
- **Result:** Base Score AUROC = 0.9307

---

### **Notebook 03: Feature Engineering**
- Adds patient-specific modifiers (age, LOS, diagnosis count)
- Detects synergistic pathologies (e.g., heart failure + pneumonia)
- Creates adjusted risk scores
- Performs feature correlation analysis

---

### **Notebook 04: Machine Learning Model Training**
- Trains 3 models: Logistic Regression, Random Forest, Gradient Boosting
- Performs hyperparameter tuning via cross-validation
- Selects optimal decision threshold (maximizes MCC)
- **Result:** Gradient Boosting AUROC = 0.9846 ⭐

---

### **Notebook 05: Model Validation**
- Compares to baseline clinical scores
- Ablation study (feature importance)
- Calibration analysis (ECE, Brier score)
- Subgroup performance (age, severity, complexity)
- **Result:** Consistent performance across demographics

---

## 🎨 Key Visualizations

All figures are in `figures/` folder:

1. **Exploratory Analysis** - Patient demographics, mortality rates by age
2. **Empirical Risks** - Distribution of diagnosis-specific risks
3. **Feature Engineering** - Correlation heatmaps, modifier distributions
4. **Model Evaluation** - ROC curves, precision-recall, confusion matrix
5. **Validation Analysis** - Calibration curves, subgroup performance

---

## 🧪 Model Details

### Approach: Hybrid Rule-Based + Machine Learning

**Phase 1: Empirical Baseline**
- Calculate historical mortality rate per ICD diagnosis code
- Add DRG severity mortality rate
- **Output:** Interpretable base score (AUROC 0.93)

**Phase 2: ML Optimization**
- Let Gradient Boosting algorithm learn optimal feature weights
- Incorporates age, LOS, diagnosis count, synergies
- **Output:** Final model (AUROC 0.98)

### Features Used (8 total)
1. `sum_diag_risk` - Sum of diagnosis-specific mortality rates (64% importance)
2. `los_days` - Length of hospital stay (17% importance)
3. `diag_count` - Number of diagnoses (16% importance)
4. `base_score` - Empirical baseline risk (2% importance)
5. `anchor_age` - Patient age (1% importance)
6. `Z` - DRG severity mortality rate (<1% importance)
7. `drg_severity` - DRG category 1-4 (<1% importance)
8. `synergy_bonus` - Dangerous diagnosis combinations (<1% importance)

### Hyperparameters (Gradient Boosting)
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 5
- `subsample`: 1.0

---

## 📈 Performance Metrics

### Discrimination
- **AUROC:** 0.9846 (95% CI: 0.983-0.986)
- **AUPRC:** 0.7160

### Classification Metrics (Threshold = 0.0197)
- **Sensitivity (Recall):** 94.9%
- **Specificity:** 93.7%
- **PPV (Precision):** 24.8%
- **NPV:** 99.9%
- **Accuracy:** 93.7%
- **F1 Score:** 0.394

### Calibration
- **Expected Calibration Error (ECE):** 0.0007 (excellent)
- **Brier Score:** 0.0103 (excellent)

### Comparison to Baselines
| Method | AUROC | Improvement |
|--------|-------|-------------|
| Age + DRG Severity | 0.8930 | Baseline |
| Empirical Base Score | 0.9306 | +4.2% |
| **Our ML Model** | **0.9846** | **+10.3%** |

---

## 🏥 Clinical Applications

### 1. Early Warning System
- Flag high-risk patients at admission (before clinical decompensation)
- Enable proactive ICU consultation, monitoring

### 2. Resource Allocation
- Guide ICU bed allocation based on objective risk
- Prioritize specialist consultations for highest-risk patients

### 3. Palliative Care
- Identify patients who may benefit from early palliative care
- Support informed goals-of-care discussions with families

### 4. Quality Metrics
- Risk-adjusted mortality for hospital benchmarking
- Fair comparison across institutions with different case mixes

---

## ⚠️ Limitations

1. **Single-center data** - MIMIC-IV is from one academic medical center (Beth Israel Deaconess)
2. **Retrospective design** - Cannot prove the model improves outcomes (need prospective trial)
3. **Missing physiological data** - No real-time vitals or labs (enables early prediction but limits accuracy)
4. **Temporal drift** - Trained on 2008-2019 data; medical practice evolves
5. **External validation needed** - Performance at other hospitals unknown

---

## 🔮 Future Work

- [ ] External validation at community hospitals, rural facilities
- [ ] Prospective randomized controlled trial
- [ ] Dynamic risk updates as hospitalization progresses
- [ ] Integration with EHR clinical decision support
- [ ] Subgroup-specific models (oncology, transplant, obstetrics)

---

## 📚 Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Visualization
- **Jupyter** - Interactive notebooks

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

**Data License:** MIMIC-IV data is governed by PhysioNet Credentialed Health Data License 1.5.0

---

## 👥 Authors

- [Your Name] - Data Science & Model Development
- [Medical Student 1] - Clinical Interpretation & Writing
- [Medical Student 2] - Literature Review & Writing

---

## 🙏 Acknowledgments

- **PhysioNet & MIT-LCP** for providing MIMIC-IV database
- **Beth Israel Deaconess Medical Center** for data collection
- [Your institution/hackathon name]

---

## 📧 Contact

For questions about this project:
- Email: [your email]
- GitHub: [your username]

---

## 📖 Citation

If you use this code or methodology, please cite:
```
Harsh, Vaishu, and Jacob. (2025). Hybrid Machine Learning Model 
for In-Hospital Mortality Prediction. GitHub repository. 
https://github.com/hduvvuru2/mortality-prediction
```

---

**⚡ Quick Start for Reviewers:**

1. Review notebooks in `notebooks/` folder (01-05)
2. View visualizations in `figures/` folder
3. Check model performance in `results/` folder
4. Read detailed documentation in `docs/` folder

**Note:** This repository contains code and results only. Data files require separate PhysioNet access.
```

3. **Save as:**
   - File name: `README.md`
   - Save type: "All Files"
   - Save location: Your project root folder

---

### **C. Create `requirements.txt` file**
```
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
jupyter==1.0.0