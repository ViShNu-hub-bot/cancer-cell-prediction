import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Cancer Cell Classifier")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load dataset
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
features = data['data']

# Streamlit UI
st.title('Breast Cancer Cell Classification')
st.sidebar.header('User Input')

# User input for test size and model selection
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.33, 0.01)
model_name = st.sidebar.selectbox('Select Model', ['XGBoost', 'Random Forest'])

# Split data and apply scaling
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)

# Model selection with hyperparameter tuning
if model_name == 'XGBoost':
    params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
    model = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), params, cv=3)
elif model_name == 'Random Forest':
    params = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15]}
    model = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3)

# Train model
model.fit(train, train_labels)
model = model.best_estimator_

# Make predictions
predictions = model.predict(test)

# Accuracy and model performance
accuracy = accuracy_score(test_labels, predictions)
st.sidebar.subheader('Model Performance')
st.sidebar.write(f'Accuracy: {accuracy:.2f}')

# Classification Report
st.subheader('Classification Report')
report = classification_report(test_labels, predictions, target_names=label_names, output_dict=True)

# Display the classification report with custom styling (black text)
report_df = pd.DataFrame(report).transpose()

# Convert the dataframe to an HTML table and add styling for black text
html_report = report_df.to_html(classes='table table-bordered table-striped', escape=False)
html_report = html_report.replace('<table', '<table style="color: black;"')

st.markdown(html_report, unsafe_allow_html=True)

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(test_labels, predictions)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Prediction Distribution with Data Labels
st.subheader('Prediction Distribution')
correct_pred = (test_labels == predictions)
pred_dist = pd.DataFrame({'Predictions': ['Correct', 'Incorrect'], 'Count': [correct_pred.sum(), len(test_labels) - correct_pred.sum()]})

fig, ax = plt.subplots()
bars = ax.bar(pred_dist['Predictions'], pred_dist['Count'])
ax.set_title('Prediction Distribution')
ax.set_ylabel('Count')
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
st.pyplot(fig)

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

