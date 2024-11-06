import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Cancer Cell Classifier")

# Load dataset
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Streamlit UI
st.title('Breast Cancer Cell Classification')
st.sidebar.header('User Input')

test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.33, 0.01)
model_name = st.sidebar.selectbox('Select Model', ['XGBoost', 'Random Forest'])

# Split data
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)

# Model selection with hyperparameter tuning
if model_name == 'XGBoost':
    params = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150]
    }
    model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), params, cv=5)
elif model_name == 'Random Forest':
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    model = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)

# Train model
model.fit(train, train_labels)

# Best parameters and model
st.sidebar.subheader('Best Model Parameters')
st.sidebar.write(model.best_params_)

# Make predictions
predictions = model.predict(test)

# Accuracy
accuracy = accuracy_score(test_labels, predictions)

# Sidebar: Model Performance
st.sidebar.subheader('Model Performance')
st.sidebar.write(f'Accuracy: {accuracy:.2f}')

# Classification Report
st.subheader('Classification Report')
report = classification_report(test_labels, predictions, target_names=label_names, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader('Confusion Matrix')
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(test_labels, predictions), annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Prediction Distribution
st.subheader('Prediction Distribution')
correct_pred = (test_labels == predictions)
pred_dist = pd.DataFrame({'Predictions': ['Correct', 'Incorrect'], 'Count': [correct_pred.sum(), len(test_labels) - correct_pred.sum()]})
st.bar_chart(pred_dist.set_index('Predictions'))

# Class Distribution
st.subheader('Class Distribution')
class_dist = pd.Series(test_labels).value_counts()
fig, ax = plt.subplots()
ax.pie(class_dist, labels=label_names, autopct='%1.1f%%')
ax.set_title('Class Distribution')
st.pyplot(fig)

# Developer Info
st.sidebar.subheader('Developed by:')
st.sidebar.write('Vishnukanth K')
st.sidebar.write('[LinkedIn](https://www.linkedin.com/in/vishnukanth-k-a5552327b/)')
