# Decision Intelligence System for Student Risk Analysis

This project is an **explainable decision intelligence system** designed to analyze student performance data, predict academic risk levels, explain contributing factors, and suggest targeted interventions.

Unlike simple prediction models, this system focuses on **decision-making and explainability**, which are critical aspects of responsible AI systems.

---

## 🎯 Project Objective

The goal of this project is to answer three key questions:

1. Is a student academically at risk?
2. Why is the student at this level of risk?
3. What actions can be taken to improve the student’s outcome?

The system classifies students into **Low**, **Medium**, or **High** risk categories and provides explanations and recommendations.

---

## 🧠 Key Features

- Risk level prediction using an explainable machine learning model
- Identification of the most influential factors behind each decision
- Actionable suggestions based on student inputs
- Interactive command-line interface for real-time analysis

---

## 📊 Dataset Description

Each row in the dataset represents a student’s academic snapshot.

### Features used:
- Attendance percentage  
- Average study hours per day  
- Previous academic marks  
- Number of assignments completed  
- Recent test score  
- Participation level (Low / Medium / High)

### Target:
- Academic risk level (Low / Medium / High)

The dataset is **simulated but realistic**, modeled after real academic indicators used in schools. The focus of the project is decision logic and explainability rather than dataset size.

---

## ⚙️ Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Decision Tree Classifier  

A decision tree model was chosen to prioritize **interpretability** over black-box accuracy.

---

## 🧩 Project Structure

