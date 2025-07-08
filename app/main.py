import streamlit as st                  # type: ignore
import pickle
import pandas as pd                    # type: ignore
import plotly.graph_objects as go      # type: ignore
import numpy as np                     # type: ignore
import os


# ---------------------- Data ------------------------
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# ---------------------- Sidebar ------------------------
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        min_val = float(data[key].quantile(0.05))
        max_val = float(data[key].quantile(0.95))
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=float(data[key].mean())
        )

    return input_dict


# ---------------------- Scale Input ------------------------
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


# ---------------------- Radar Chart ------------------------
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
           input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )

    return fig


# ---------------------- Prediction ------------------------
def add_predictions(input_data):
    try:
        model = pickle.load(open("model/model.pkl", "rb"))
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
    except FileNotFoundError:
        st.error("Model or scaler file not found. Make sure 'model.pkl' and 'scaler.pkl' are in 'model/' folder.")
        st.stop()

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    proba = model.predict_proba(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    if prediction[0] == 0:
        st.success("✅ The cell cluster is **Benign**")
    else:
        st.error("⚠️ The cell cluster is **Malignant**")

    st.write(f"**Probability of being Benign:** {proba[0][0]:.2f}")
    st.write(f"**Probability of being Malignant:** {proba[0][1]:.2f}")

    # Save to session state
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({**input_data, "Prediction": "Benign" if prediction[0] == 0 else "Malignant"})


# ---------------------- Main App ------------------------
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.markdown("""
        <div style="background-color:#02475e;padding:20px;border-radius:10px">
        <h2 style="color:white;text-align:center;">🧬 Breast Cancer Prediction Dashboard</h2>
        <p style="color:white;text-align:center;">An AI-powered decision support system for medical professionals</p>
        </div>
    """, unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts whether a breast mass is benign or malignant using cytology measurements.")

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        add_predictions(input_data)

    # Model metrics (example - adjust as per your model)
    st.markdown("### 📊 Model Evaluation")
    st.metric("Accuracy", "97.5%")
    st.metric("AUC Score", "0.985")

    # Prediction history
    if "history" in st.session_state:
        st.markdown("### 🕒 Recent Predictions")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df.tail(5))

    # Footer
    st.markdown("""
        <hr style="border-top: 1px solid #bbb;">
        <p style="text-align:center; font-size: 14px;">
            Built with ❤️ by Azaam | Powered by Scikit-Learn, Plotly, and Streamlit.
        </p>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
