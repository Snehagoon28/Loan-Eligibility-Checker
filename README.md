This project is a loan eligibility prediction tool I built using machine learning techniques. It analyzes applicant data, identifies key factors influencing loan approval, and predicts eligibility. The code is structured to handle real-world datasets, preprocess data, and provide insights into approval trends.

Features:
Data Preprocessing:

Missing values in numerical columns are filled with the mean, and categorical columns are filled with the most frequent value.
Categorical variables are encoded for compatibility with machine learning models.
Machine Learning Model:

A Random Forest Classifier is used to predict loan eligibility.
The dataset is split into training and testing sets to ensure robust evaluation.
Performance Evaluation:

The script outputs model accuracy and a detailed classification report.
Feature importance is calculated to highlight the most significant factors influencing loan approval.
Insights and Visualizations:

Profiles most likely to get loan approval are summarized based on trends in the data.
A bar chart shows feature importance, providing a visual understanding of key factors.

Results:
Model Accuracy: Around 70% on the test dataset.
Feature Importance Visualization: Clearly indicates which factors matter the most.
Approval Trends: Highlights characteristics of applicants with higher chances of loan approval.
