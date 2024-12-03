import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px


def load_data():
    data = pickle.load(open('data.pkl','rb'))
    return data

data = load_data()

def creat_sidebar():
    st.sidebar.header("Patient Reports Information")
    
    sidebar_labels = [
    ("Chest Pain Type (cp)", "cp"),
    ("Resting Blood Pressure (trestbps)", "trestbps"),
    ("Cholesterol (chol)", "chol"),
    ("Resting ECG Results (restecg)", "restecg"),
    ("Maximum Heart Rate Achieved (thalach)", "thalach"),
    ("ST Depression (oldpeak)", "oldpeak"),
    ("Slope of ST Segment (slope)", "slope"),
    ("Number of Major Vessels (ca)", "ca"),
    ("Thalassemia (thal)", "thal")]

    input_dic = {}
    
    # age numeric input
    input_dic['age'] = st.sidebar.number_input("Age",value= int(data['age'].min()))
    
    # radio input for gender 
    input_dic['sex'] = st.sidebar.radio("Gender",["Male","Female"])
    input_dic['sex'] = 1 if input_dic['sex'] == "Male" else 0
    
    # slidebar input for remaining features
    for label, key in sidebar_labels:
        input_dic[key]= st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value= float(data[key].mean())
        )
    return input_dic
        
def predictions(input_val):
    model = pickle.load(open('model.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))

    input_arry = np.array(list(input_val.values())).reshape(1,-1)
    input_scaled = scaler.transform(input_arry)
    pred = model.predict(input_scaled)

    
    st.subheader('Heart disease prediction')
    st.write("The Patient is: ")

    if pred == 0:
        st.write("<span class= 'diagnosis normal'>Normal</span>", unsafe_allow_html=True) 
    else:
        st.write("<span class= 'diagnosis patient'>Heart Patient</span>",unsafe_allow_html=True)

    st.write(f'Model Accuracy is: 98.5%')
    st.write("##### Note:")
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def creat_graph(input_val):
    scaler = pickle.load(open('scaler.pkl','rb'))

    input_arry_val = np.array(list(input_val.values())).reshape(1,-1)
    x = scaler.transform(input_arry_val)
    values= x.flatten()
    keys = np.array(list(input_val.keys()))

    fig = px.line(x=keys, y=values, markers=True, title='Reports Graph')

    fig.update_layout(
    title_font_size=20,
    # margin=dict(l=20, r=20, t=30, b=20),
    # width=800,
    # height=400,

    xaxis=dict(showgrid=True,
        title=dict(
            text="Reports",  # X-axis title
            font=dict(size=18, color="black", weight="bold"),  # Title font
        ),
        tickfont=dict(size=14, color="gray", weight="bold"),  # Axis tick font
    ),
    yaxis=dict(
        title=dict(
            text="Scaled Values",  # Y-axis title
            font=dict(size=18, color="black", weight="bold"),  # Title font
        ),
        tickfont=dict(size=14, color="gray", weight="bold"),  # Axis tick font
    ))

    return fig

def main():
    # Add a title
    st.set_page_config(page_title="Heart Disease Diagnosis",
                    page_icon="🫀", 
                    layout="wide", 
                    initial_sidebar_state="expanded")
        
    input_val= creat_sidebar()

    with open('style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    with st.container():
        st.title("Heart Disease Diagnosis")
        st.write("Please connect this app to your lab reports to help diagnose heart disease. This app predicts using a machine learning model whether a patient has a heart disease or not, based on his lab reports. You can also update the measurements by hand using the sliders in the sidebar.")
        col1, col2 = st.columns([4,1])
        with col1:
           plot= creat_graph(input_val)
           st.plotly_chart(plot)
        with col2:
            predictions(input_val)


if __name__ == '__main__':
    main()
