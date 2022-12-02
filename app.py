import joblib
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

cancer_model=pickle.load(open('cancer_model.pkl','rb'))
diabetes_model=joblib.load(open('diabetes_model.pkl','rb'))
heart_model=joblib.load(open('heart_model.pkl','rb'))
kidney_model=joblib.load(open('kidney_model.pkl','rb'))
liver_model=joblib.load(open('liver_model.pkl','rb'))
parkinsons_model=joblib.load(open('parkinsons_model.pkl','rb'))

# sidebar

with st.sidebar:
    selected=option_menu('Multiple Disease Detection System',['Welcome',
        'Cancer','Diabetes','Heart','Kidney','Liver','Parkinsons'],
        icons=['book','bookmark-check','bi-app-indicator','heart','file-bar-graph','activity','bi-person'],
        default_index=0)

if selected == 'Welcome':
    st.title('Welcome to Automatic Disease Detection Using Machine Learning')
    st.image('disease-diagnosis-using-machine-learning.png')

if selected=='Cancer':
    st.title('Breast Cancer Test')
    concave=st.number_input('Concave',value=0.012,step=0.025)
    area=st.number_input('Area',value=500.0,step=0.1)
    radius=st.number_input('Radius',value=10.0,step=0.025)
    perimeter=st.number_input('Perimeter',value=60.0,step=0.1)
    concavity=st.number_input('Concavity',value=0.020470, step=0.025)

    result=''
    if st.button('Get result'):
        cancer_result=cancer_model.predict([[concave,area,radius,perimeter,concavity]])
        if cancer_result[0]==1:
            result='You are likely to have cancer. Please see a doctor.'
        else:
            result='You do not have cancer.'
    st.success(result)

if selected=='Diabetes':
    st.title('Diabetes Test')
    pregnancies=st.number_input('Number of pregnencies',min_value=0,max_value=10,step=1,value=1)
    glucose=st.number_input('Glucose Level',min_value=50,step=1,value=100)
    bp=st.number_input('Current blood Pressure',min_value=50,max_value=130,step=1,value=70)
    bmi=st.number_input('BMI',min_value=1,max_value=70,value=30,step=1)
    pedigree=st.number_input('Diabetes Pedigree Function',min_value=0.2,max_value=2.5,step=0.1,value=0.5)
    age=st.number_input('Age',min_value=0,max_value=100,step=1,value=18)

    result = ''
    if st.button('Get result'):
        diabetes_result = diabetes_model.predict([[pregnancies,glucose,bp,bmi,pedigree,age]])
        if diabetes_result[0] == 1:
            result = 'You are likely to have diabetes. Please see a doctor.'
        else:
            result = 'You do not have diabetes.'
    st.success(result)

if selected=='Heart':
    st.title('Heart Disease Test')
    chest_pain=st.selectbox('Chest Pain type',['Typical Angina','Atypical Angina', 'Non-Anginal Pain','Asymptomatic'])
    cp=int()
    if chest_pain=='Typical Angina':
        cp=0
    elif chest_pain=='Atypical Angina':
        cp=1
    elif chest_pain=='Non-Anginal Pain':
        cp=2
    else:
        cp=3
    rest_bp=st.number_input('Resting Blood Pressure (in mm of Hg)',value=120,step=1)
    cholestrol=st.number_input('Serum Cholestrol (in mg/dl)',value=200, step=5)
    blood_sugar=st.radio('Is Fasting Blood Sugar <120 mg/dl',['Yes','No'])
    ecg=st.selectbox('Electrocardiograph Result',['Normal','Having ST-T wave Abnormality','Showing propbable or definate left Ventricular Hypertrophy'])
    max_heart_rate=st.number_input('Maximum Heart Rate Achieved',value=150,step=1)
    exercise=st.selectbox('Exercise Induced Angina',['Yes','No'])

    result = ''
    if st.button('Get result'):
        heart_result = heart_model.predict([[cp,rest_bp,cholestrol,1 if blood_sugar=='Yes' else 0,0 if ecg=='Normal' else 1,max_heart_rate,1 if exercise=='Yes' else 0]])
        if heart_result[0] == 1:
            result = 'You are likely to have a heart disease. Please see a doctor.'
        else:
            result = 'Your heart is healthy.'
    st.success(result)

if selected=='Kidney':
    st.title('Kidney Test')
    bp=st.number_input('Blood Pressure',value=50,step=1)
    gravity=st.number_input('Specific Gravity',value=1.000,step=0.025)
    albumin=st.number_input('Albumin',value=1.0,step=0.5)
    sugar=st.number_input('Blood Sugar Level',value=1,step=1,max_value=5)
    rbc=st.radio('Red Blood Cells Count',['abnormal','normal'])
    pbc=st.radio('Pus Cell Count',['abnormal','normal'])
    pcclumps=st.radio('Pus Cell Clumps',['present','not present'])

    result = ''
    if st.button('Get result'):
        kidney_result = kidney_model.predict([[bp,gravity,albumin,sugar,1 if rbc=='abnormal' else 0,1 if pbc=='abnormal' else 1,1 if pcclumps=='present' else 0]])
        if kidney_result[0] == 1:
            result = 'You are likely to have a kidney disease. Please see a doctor.'
        else:
            result = 'Your kidney is healthy.'
    st.success(result)


if selected=='Liver':
    st.title('Liver Test')
    total_bilirubin=st.number_input('Total Bilirubin',value=0.4,step=0.1)
    direct_bilirubin=st.number_input('Direct Bilirubin',value=0.1,step=0.1)
    alkeline=st.number_input('Alkaline Phosphotase',value=70,step=1)
    alamine=st.number_input('Alamine Aminotransferase',value=10,step=1)
    protein=st.number_input('Total Protein',value=3.0,step=0.1)
    albumin=st.number_input('Albumin',value=2.7,step=0.1)
    agratio=st.number_input('Albumin to Globulin ratio',value=3.9,step=0.1)


    result = ''
    if st.button('Get result'):
        liver_result = liver_model.predict([[total_bilirubin, direct_bilirubin, alkeline, alamine, protein, albumin, agratio]])
        if liver_result[0] == 1:
            result = 'You are likely to have a liver disease. Please see a doctor.'
        else:
            result = 'Your liver is healthy.'
    st.success(result)


if selected=='Parkinsons':
    st.title("Parkinson's Disease Test")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)',value=150.0,step=0.25)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)',value=160.0,step=0.5)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)',value=100.0,step=1.0)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)',value=0.005582,step=0.0005)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)',value=0.000038,step=0.00001)

    with col1:
        RAP = st.number_input('MDVP:RAP',value=0.002868,step=0.001)

    with col2:
        PPQ = st.number_input('MDVP:PPQ',value=0.00315, step=0.001)

    with col3:
        DDP = st.number_input('Jitter:DDP',value=0.008602,step=0.001)

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer',value=0.027968,step=0.0025)

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)',value=0.259400,step=0.05)

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3',value=0.01388,step=0.01)

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5',value=0.016510,step=0.01)

    with col3:
        APQ = st.number_input('MDVP:APQ',value=0.023982,step=0.01)

    with col4:
        DDA = st.number_input('Shimmer:DDA',value=0.041660,step=0.01)

    with col5:
        NHR = st.number_input('NHR',value=0.011348,step=0.01)

    with col1:
        HNR = st.number_input('HNR',value=22.14200,step=1.0)

    with col2:
        RPDE = st.number_input('RPDE',value=0.516489,step=0.1)

    with col3:
        DFA = st.number_input('DFA',value=0.743321,step=0.1)

    with col4:
        spread1 = st.number_input('spread1',value=-5.706983,step=0.5)

    with col5:
        spread2 = st.number_input('spread2',value=0.188637,step=0.1)

    with col1:
        D2 = st.number_input('D2',value=2.170480,step=0.25)

    with col2:
        PPE = st.number_input('PPE',value=0.210302,step=0.025)

    # code for Prediction
    result = ''

    # creating a button for Prediction
    if st.button("Get Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            result = "You are likely to have Parkinson's disease. Please see a doctor"
        else:
            result = "You do not have Parkinson's disease"

    st.success(result)


