import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Cancer Cell Classification")
# Suppress the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Breast Cancer dataset
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Create a Streamlit UI for user input
st.title('Breast Cancer Cell Classification')
st.sidebar.header('User Input')

# Sidebar input for test size and model selection
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.33, 0.01)
model_name = st.sidebar.selectbox('Select Model', ['Naive Bayes', 'Random Forest'])

# Split the dataset into train and test sets based on user input
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)

# Model Training and Prediction
if model_name == 'Naive Bayes':
    model = GaussianNB()
elif model_name == 'Random Forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(train, train_labels)

# Predictions
predictions = model.predict(test)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)

# Display accuracy
st.sidebar.subheader('Model Performance')
st.sidebar.write(f'Accuracy: {accuracy:.2f}')

# Classification Report
st.subheader('Classification Report')
report = classification_report(test_labels, predictions, target_names=label_names, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()

# Prediction Distribution
st.subheader('Prediction Distribution')
correct_pred = (test_labels == predictions)
pred_dist = pd.DataFrame({'Predictions': ['Correct', 'Incorrect'], 'Count': [correct_pred.sum(), len(test_labels) - correct_pred.sum()]})
st.bar_chart(pred_dist.set_index('Predictions'))

# Class Distribution
st.subheader('Class Distribution')
class_dist = pd.Series(test_labels).value_counts()
plt.pie(class_dist, labels=label_names, autopct='%1.1f%%')
plt.title('Class Distribution')
st.pyplot(plt.gcf(), use_container_width=True)

# Developed by
st.sidebar.subheader('Developed by:')
st.sidebar.write('Vishnukanth K')
st.sidebar.write('[LinkedIn](https://www.linkedin.com/in/vishnukanth-k-a5552327b/)')
