


#################################
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle

# # Page configuration
# st.set_page_config(page_title="Health Care Center", page_icon="🩺", layout="centered")
# st.markdown("<h1 style='text-align: center; color: green;'>Health Care Center 🩺</h1>", unsafe_allow_html=True)

# # Background Image Style

# st.markdown(
#     """
#    <style>
#     .stApp {
#         color: white; /* White text */
#     }
#     .title {
#         font-size: 24px;
#         font-weight: bold;
#         color: white; /* White text color for the title */
#         text-align: center;
#         margin-bottom: 15px;
#     }
#     .predict-btn {
#         background-color: #1976D2;  /* Lighter blue button */
#         color: white;
#         border-radius: 5px;
#         padding: 10px 20px;
#         font-size: 18px;
#         cursor: pointer;
#     }
#     .predict-btn:hover {
#         background-color: #1565C0; /* Darker blue when hovered */
#     }
#     .st-expanderHeader {
#         color: white;  /* White text for expander headers */
#         font-size: 18px;
#         font-weight: bold;
#     }
#     .st-expanderContent {
#         color: white;  /* White text for content */
#         font-size: 16px;
#     }
#     .expander-header {
#         font-size: 18px;
#         color: white;  /* White text for expander headers */
#     }
#     .expander-content {
#         font-size: 16px;
#         color: white;  /* White text for expander content */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # Load datasets and model
# sym_des = pd.read_csv("datasets/symtoms_df.csv")
# precautions = pd.read_csv("datasets/precautions_df.csv")
# workout = pd.read_csv("datasets/workout_df.csv")
# description = pd.read_csv("datasets/description.csv")
# medications = pd.read_csv("datasets/medications.csv")
# diets = pd.read_csv("datasets/diets.csv")
# svc = pickle.load(open('models/svc.pkl', 'rb'))

# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# # Helper function to retrieve additional information
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description'].values
#     desc = " ".join(desc) if desc.size > 0 else "No description available."
#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten() if not precautions.empty else ["No precautions available."]
#     med = medications[medications['Disease'] == dis]['Medication'].values if not medications.empty else ["No medications available."]
#     die = diets[diets['Disease'] == dis]['Diet'].values if not diets.empty else ["No recommended diets available."]
#     wrkout = workout[workout['disease'] == dis]['workout'].values if workout.size > 0 else ["No workout suggestions available."]
#     return desc, pre, med, die, wrkout

# # Model Prediction function
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for symptom in patient_symptoms:
#         if symptom in symptoms_dict:
#             input_vector[symptoms_dict[symptom]] = 1
#     predicted_index = svc.predict([input_vector])[0]
#     return diseases_list.get(predicted_index, "Unknown Disease")

# # Streamlit app structure
# st.subheader("Enter Symptoms for Diagnosis")
# symptoms = st.text_input("Enter symptoms (comma-separated)", placeholder="Type symptoms such as itching, fever, etc.")

# if st.button("Predict", key="predict-btn"):
#     if symptoms:
#         user_symptoms = [s.strip().lower() for s in symptoms.split(',')]
#         predicted_disease = get_predicted_value(user_symptoms)
        
#         # Display predicted disease
#         st.markdown(f"<div class='title'>**Predicted Disease:** {predicted_disease}</div>", unsafe_allow_html=True)
        
#         # Fetch additional information
#         dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
#         # Display disease details in expander sections
#         with st.expander("📝 Description", expanded=True):
#             st.markdown(f"<div class='expander-header'>{dis_des}</div>", unsafe_allow_html=True)

#         # with st.expander("🚧 Precautions"):
#         #     st.markdown(f"<div class='expander-header'>Precautions</div>", unsafe_allow_html=True)
#         #     for pre in precautions:
#         #         st.write(f"<div class='expander-content'>{pre}</div>", unsafe_allow_html=True)

#         with st.expander("🚧 Precautions"):
#             st.markdown(f"<div class='expander-header'>Precautions</div>", unsafe_allow_html=True)
#             for pre in precautions:
#                 st.markdown(f"<li style='color: white;'>{pre}</li>", unsafe_allow_html=True)


#         with st.expander("💊 Medications"):
#             st.markdown(f"<div class='expander-header'>Medications</div>", unsafe_allow_html=True)
#             for med in medications:
#                 st.write(f"<div class='expander-content'>{med}</div>", unsafe_allow_html=True)

#         with st.expander("🥗 Diet Recommendations"):
#             st.markdown(f"<div class='expander-header'>Diet Recommendations</div>", unsafe_allow_html=True)
#             for diet in rec_diet:
#                 st.write(f"<div class='expander-content'>{diet}</div>", unsafe_allow_html=True)

#         # with st.expander("🏋️ Workout Suggestions"):
#         #     st.markdown(f"<div class='expander-header'>Workout Suggestions</div>", unsafe_allow_html=True)
#         #     for work in workout:
#         #         st.write(f"<div class='expander-content'>{work}</div>", unsafe_allow_html=True)

#         with st.expander("🏋️ Workout Suggestions"):
#             st.markdown(f"<div class='expander-header'>Workout Suggestions</div>", unsafe_allow_html=True)
#             for work in workout:
#                 st.markdown(f"<li style='color: white;'>{work}</li>", unsafe_allow_html=True)






import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Health Care Center", page_icon="🩺", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>Health Care Center 🩺</h1>", unsafe_allow_html=True)

# Load datasets and model
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
svc = pickle.load(open('models/svc.pkl', 'rb'))

# List of symptoms for auto-suggestion
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
symptoms_list = list(symptoms_dict.keys())
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Helper function to retrieve additional information
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values
    desc = " ".join(desc) if desc.size > 0 else "No description available."
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten() if not precautions.empty else ["No precautions available."]
    med = medications[medications['Disease'] == dis]['Medication'].values if not medications.empty else ["No medications available."]
    die = diets[diets['Disease'] == dis]['Diet'].values if not diets.empty else ["No recommended diets available."]
    wrkout = workout[workout['disease'] == dis]['workout'].values if workout.size > 0 else ["No workout suggestions available."]
    return desc, pre, med, die, wrkout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    predicted_index = svc.predict([input_vector])[0]
    return diseases_list.get(predicted_index, "Unknown Disease")

# Streamlit app structure
st.subheader("Enter Symptoms for Diagnosis")
selected_symptoms = st.multiselect("Select symptoms", options=symptoms_list, help="Type to search symptoms")

if st.button("Predict", key="predict-btn"):
    if selected_symptoms:
        predicted_disease = get_predicted_value(selected_symptoms)
        
        # Display predicted disease
        st.markdown(f"<div class='title'>**Predicted Disease:** {predicted_disease}</div>", unsafe_allow_html=True)
        
        # Fetch additional information
        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
        
        # Display disease details in expander sections
        with st.expander("📝 Description", expanded=True):
            st.markdown(f"<div class='expander-header'>{dis_des}</div>", unsafe_allow_html=True)

        with st.expander("🚧 Precautions"):
            for pre in precautions:
                st.write(f"- {pre}", unsafe_allow_html=True)

        with st.expander("💊 Medications"):
            for med in medications:
                st.write(f"- {med}", unsafe_allow_html=True)

        with st.expander("🥗 Diet Recommendations"):
            for diet in rec_diet:
                st.write(f"- {diet}", unsafe_allow_html=True)

        with st.expander("🏋️ Workout Suggestions"):
            for work in workout:
                st.write(f"- {work}", unsafe_allow_html=True)
    else:
        st.error("Please select at least one symptom.")
