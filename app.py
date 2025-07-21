import streamlit as st
import requests

st.title("Finger Sign Detection App")

uploaded_file = st.file_uploader("Upload a hand image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Fingers"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        try:
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Detected: {result['fingers_detected']} finger(s)")
            else:
                st.error(f" Detection failed. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
