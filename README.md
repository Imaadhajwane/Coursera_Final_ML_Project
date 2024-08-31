# Customer Churn Prediction Project

Welcome to the Customer Churn Prediction Project! This project aims to develop a machine learning model to predict which customers are likely to cancel their subscription in the next period. Below is a detailed overview of the project's structure and steps.

## 1. Opportunity Evaluation

### Project Topic

**Problem Statement:** Customer Churn Prediction for a Subscription-Based Service

**Context:** Companies offering subscription-based services often struggle with customer churn. Predicting which customers are likely to churn can help these companies retain valuable customers.

**Why ML?**
- **Data Availability:** Subscription services collect large amounts of customer data, including usage patterns, payment history, and support interactions.
- **Feasibility:** Predictive models can identify patterns indicative of future churn.
- **Business Impact:** Reducing churn can significantly improve profitability as acquiring new customers is often more expensive than retaining existing ones.

### Dataset Selection

The dataset used for this project is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. It includes customer information such as usage statistics, account details, and churn status.

## 2. CRISP-DM Business Understanding

### Defining the Problem

- **Business Objective:** Reduce customer churn by predicting which customers are likely to cancel their subscription in the next period.
- **Data Mining Objective:** Develop a predictive model to identify customers who are likely to churn based on their historical data.

### Success Criteria

- **Outcome Metrics:** Reduction in churn rate by a specific percentage (e.g., 5% reduction).
- **Output Metrics:** Accuracy, precision, recall, and F1-score of the churn prediction model.

### Identifying Relevant Factors

- **Customer Demographics:** Age, gender, income level.
- **Service Usage:** Number of services used, frequency of use.
- **Account Information:** Contract type, tenure, payment method.
- **Support Interaction:** Number of support tickets, type of issues raised.

## 3. Solution Validation Plan

### Solution Concept

The solution will be a binary classification model predicting customer churn (yes/no) based on provided features. Models like logistic regression, decision trees, or more complex methods like Random Forest or Gradient Boosting will be used.

### Validation Strategy

1. **Data Splitting:** Split the dataset into training, validation, and test sets.
2. **Iterative Experimentation:**
   - **Initial Model:** Train a basic model (e.g., logistic regression) and evaluate its performance.
   - **Feature Engineering:** Add or remove features, and transform data as needed.
   - **Model Tuning:** Optimize hyperparameters using cross-validation.
   - **Model Comparison:** Compare different models and select the best one.
   - **User Feedback:** Test the model predictions on a small group of real customers and gather feedback.

## 4. ML System Design

### Key System-Level Architecture Decisions

- **Data Pipeline:**
  - **Ingestion:** Collect real-time customer data.
  - **Processing:** Clean and preprocess the data (e.g., handle missing values, standardize inputs).
  - **Feature Engineering:** Generate additional features (e.g., time since last interaction).
  
- **Model Deployment:**
  - **Batch Processing:** Predict churn probability on a scheduled basis (e.g., weekly).
  - **Real-Time Scoring:** Predict churn probability in real-time as new data comes in.
  
- **Monitoring:**
  - **Performance Monitoring:** Track the model's accuracy, precision, recall, etc., over time.
  - **Data Drift Detection:** Monitor changes in input data distributions that could affect model performance.

## 5. Potential Risks in Production

### Identifying Risks

- **Data Drift:** The model may become less accurate if customer behavior changes over time. Regular retraining will be necessary.
- **Concept Drift:** The model may need updating if the factors influencing churn change.
- **Training-Serving Skew:** Differences between training data and live data could degrade performance. Ensure consistent preprocessing.
- **Latency:** Ensure predictions are made quickly enough to be actionable if deployed in a real-time system.

## Machine Learning Algorithms Used:

  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Support Vector Machine**
  - **K-Nearest Neighbors**
  - **Naive Bayes**
  - **Gradient Boosting**

## Accuracy Result:

```
	Models	                    Accuracy
- Logistic Regression	        0.819730
- Decision Tree	                0.713982
- Random Forest	                0.793471
- Support Vector Machine	0.806955
- K-Nearest Neighbors	        0.756565
- Naive Bayes	                0.757984
- Gradient Boosting    	        0.810504
```

## Video Execution:

- **URL**:  [Final_Code_Execution_with_Description.](https://drive.google.com/drive/folders/1k7aWZi9SoGHsdCJvOcUkWpLZmxvnKn72?usp=drive_link)


