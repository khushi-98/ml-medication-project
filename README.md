# Disease Prediction System
Overview
The Disease Prediction System is a machine learning-powered application designed to predict potential diseases based on user-provided symptoms. Built using Python, the system incorporates data preprocessing, model training, user interaction, and auxiliary data integration to provide comprehensive and accurate predictions. The web-based interface, developed using Streamlit, ensures cross-platform accessibility and ease of use.


Features
1.Data Preprocessing:

Handles missing values.
Encodes categorical variables using label encoding for numerical compatibility.
Splits the dataset into training and testing subsets (70-80% training, 20-30% testing).'

2.Machine Learning:

Trains multiple classifiers: Support Vector Machines (SVM), Random Forest, K-Nearest Neighbors (KNN).
Evaluates models using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
Selects the best-performing model (SVM).

3.Prediction System:

Accepts symptoms as user input.
Uses the serialized SVM model to predict diseases.

4.Supplementary Information:

Links to auxiliary data for disease descriptions, precautionary measures, medications, diets, and workouts.

5.Interactive Dashboard:

Enables users to input symptoms, view predictions, and access supplementary information.

6.System Deployment:

Hosted via GitHub as a Streamlit web application for real-time interaction.




How It Works
1. Data Preprocessing Layer:
Input: Raw dataset containing symptoms and disease labels.
Processes: Missing value handling, label encoding, data splitting.
2. Model Selection and Training Layer:
Input: Preprocessed training data.
Processes: Model training and evaluation.
Output: Best-performing model (SVM) serialized using pickle.
3. Model Deployment Layer:
Input: User-provided symptoms.
Processes: Maps symptoms to a vectorized format and predicts diseases using the SVM model.
4. Auxiliary Data Integration Layer:
Maps predicted diseases to datasets containing detailed descriptions, precautions, medications, and other recommendations.
5. Interactive User Interface Layer:
Input: Symptoms provided via a text box or dropdown menu.
Output: Predicted disease and supplementary information.



System Requirements
1. Python: 3.8+
2. Libraries:
3. Pandas
4. NumPy
5. Scikit-learn
6. Streamlit
7. Pickle





