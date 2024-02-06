import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Cancer Cell Classifier")
# Suppress the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)


data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Create a Streamlit UI for user input
st.title('Breast Cancer Cell Classification')
st.sidebar.header('User Input')


test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.33, 0.01)
model_name = st.sidebar.selectbox('Select Model', ['Naive Bayes', 'Random Forest'])


train, test, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=42)


if model_name == 'Naive Bayes':
    model = GaussianNB()
elif model_name == 'Random Forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(train, train_labels)


predictions = model.predict(test)


accuracy = accuracy_score(test_labels, predictions)


st.sidebar.subheader('Model Performance')
st.sidebar.write(f'Accuracy: {accuracy:.2f}')


st.subheader('Classification Report')
report = classification_report(test_labels, predictions, target_names=label_names, output_dict=True)
st.write(pd.DataFrame(report).transpose())


st.subheader('Confusion Matrix')
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()


st.subheader('Prediction Distribution')
correct_pred = (test_labels == predictions)
pred_dist = pd.DataFrame({'Predictions': ['Correct', 'Incorrect'], 'Count': [correct_pred.sum(), len(test_labels) - correct_pred.sum()]})
st.bar_chart(pred_dist.set_index('Predictions'))


st.subheader('Class Distribution')
class_dist = pd.Series(test_labels).value_counts()
plt.pie(class_dist, labels=label_names, autopct='%1.1f%%')
plt.title('Class Distribution')
st.pyplot(plt.gcf(), use_container_width=True)


st.sidebar.subheader('Developed by:')
st.sidebar.write('Vishnukanth K')
st.sidebar.write('[LinkedIn](https://www.linkedin.com/in/vishnukanth-k-a5552327b/)')
