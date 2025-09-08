
# Academic Performance Prediction using TaBERT and XAI

This repository implements an end-to-end pipeline for predicting student academic performance and mental health outcomes using **classical ML models, ensemble methods, deep models (LSTM), and a TabTransformer (TaBERT-style)**.  
It integrates **feature selection techniques (IG, Gain Ratio, Entropy, Gini Index)**, **explainability (LIME, SHAP)**, and **statistical analysis** for robust evaluation.

---

## 📂 Project Workflow

1. **Data Load**  
   - CSV-based dataset  
   - Numerical + Categorical features  

2. **Preprocessing**  
   - Imputation (median for numeric, mode for categorical)  
   - Standard scaling (numeric)  
   - One-hot encoding (categorical)

3. **Feature Categorization**  
   - Mental Health Factors  
   - Personal Factors  
   - Social Factors  
   - Academic (CGPA-based)

4. **Feature Selection Methods**  
   - Information Gain (IG)  
   - Gain Ratio (GR)  
   - Entropy  
   - Gini Index  

5. **Models Implemented**  
   - Support Vector Machine (SVM)  
   - Logistic Regression (LR)  
   - Random Forest (RF)  
   - Gradient Boosting (GB)  
   - LSTM (tabular sequence modeling)  
   - TabTransformer (TaBERT-style dual attention for tabular data)   

6. **Experimental Blocks**  
   Separate evaluations per factor group and feature selector:  
   - Mental Health × {IG, GR, Entropy}  
   - Social × {IG, GR, Entropy}  
   - Personal × {IG, GR, Entropy}  
   - Academic Performance × {IG, GR, Entropy}  

7. **Course-wise Performance**  
   - Model evaluation by course subgroup  

8. **Explainability**  
   - **LIME**: Local instance-level feature explanations  
   - **SHAP**: Global feature importance  

9. **Statistical Tests**  
   - t-test, ANOVA, Chi-square, Z-test  
   - Summarized feature significance overview  

---

## 📊 Results Produced
- Model-wise accuracy, precision, recall, and F1  
- Feature importance (IG, GR, Entropy, Gini)  
- Subgroup (course-wise) performance tables  
- XAI outputs (LIME feature weights, SHAP values)  
- Statistical test results with significance plots  

---

## ⚙️ Requirements

Install dependencies with:


```bash
pip install pandas numpy scikit-learn scipy torch torchvision torchaudio shap lime transformers einops
```bash
pip install pandas numpy scikit-learn scipy torch torchvision torchaudio shap lime transformers einops
