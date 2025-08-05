# Diabetes-prediction-ml
# 🧠 Diabetes Prediction Using Machine Learning

This project was completed as part of the Machine Learning A course (Assignment 1: Exploring Structured Data Using ML).  
It explores the end-to-end ML pipeline to predict diabetes using health-related structured data from real-world sources.

## 📌 Objective

To develop a machine learning solution that predicts whether a person has diabetes based on health indicators.  
The work includes problem understanding, data exploration, ML modeling, evaluation, and a basic Streamlit UI for interaction.

---

## 🧩 Part A: Problem Understanding

**Problem Statement:**  
Diabetes is a chronic health condition with serious complications if undiagnosed. This project aims to predict the likelihood of diabetes based on health survey indicators, helping public health workers and doctors make early interventions.

**Stakeholders:**  
- Doctors and nurses  
- Public health analysts  
- Patients  
- Healthcare app developers

---

## 📊 Part B: Dataset Description

- **Source:** [Kaggle - BRFSS Diabetes Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Rows:** 253,680  
- **Columns:** 22 (includes BMI, Smoking, Physical Activity, Sleep, etc.)
- **Target column:** Diabetes_binary

---

## ❓ Part C: 10 Questions for Stakeholders

This project formulates 10 questions such as:
- Which indicators most strongly correlate with diabetes?
- Can we predict diabetes using only lifestyle indicators?
- How accurate can our model be in predicting diabetes?

(*Full list of questions inside the notebook.*)

---

## 🔁 Part D: Machine Learning Pipeline

- Data cleaning: Handling missing values, encoding, scaling
- EDA: Correlation matrix, pie charts, distribution analysis
- Feature engineering: Column filtering and selection
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Accuracy, Confusion Matrix, Feature Importance
- Diagrams: Workflow and Data Overview included

---

## 💡 Part E: Frontend Integration (Bonus)

This project includes a simple Streamlit app to:
- Upload user input via sliders and checkboxes
- Predict diabetes instantly using trained model (`.pkl`)

---

## 📈 Part F: Results & Conclusion

- **Best Model:** Random Forest Classifier  
- **Accuracy:** ~[insert value, e.g. 85%]  
- **Insights:** Physical activity, general health, and BMI are top predictors  
- **Recommendations:** This solution could be integrated into a screening app

---

## 🧪 Tools Used

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, seaborn  
- `streamlit`, `joblib`, xgboost  
- Visualization: seaborn heatmaps, pie chart, and matplotlib

---

## 📁 Files in This Repo

- diabetes_prediction.ipynb – Main Jupyter Notebook
- best_model.pkl – Trained ML model
- streamlit_app.py – Streamlit frontend (optional)
- diabetes_pie_chart.png – Visualization
- plot_correlation_matrix.png – Correlation heatmap
- requirements.txt – Python dependencies

---

## ✍️ Author

**Natthawan Srisan**  
MSc. Software Engineering – University of Europe for Applied Sciences  
GitHub: [github.com/NatthawanSrisan](https://github.com/NatthawanSrisan)
