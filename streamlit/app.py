import streamlit as st
import pandas as pd
import requests
import os

GRADIO_URL = "127.0.0.1:8000" # temp localhost

# CONFIG
API_URL = "http://" + GRADIO_URL + "/get-violations/"  # <-- UPDATE THIS
IMAGE_BASE_URL = "http://" + GRADIO_URL + "/get-image/"   # <-- UPDATE THIS

st.set_page_config(page_title="PPE Dashboard", layout="wide")
#st.title("🛡️ PPE Violation Dashboard")
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <img src="https://www.coca-colacompany.com/content/dam/company/us/en/the-coca-cola-company-logo-white.svg" 
             alt="Coca-Cola Logo" style="height: 60px; margin-left: 20px;">
         <h1 style="margin: 0;">🛡️ PPE Violation Dashboard</h1>
    </div>
    <hr style="margin-top: 0.5em; margin-bottom: 1em;">
    """,
    unsafe_allow_html=True
)

# Fetch from FastAPI
@st.cache_data(ttl=30)
def fetch_data():
    response = requests.get(API_URL)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")
    return pd.DataFrame(response.json())

try:
    df = fetch_data()
except Exception as e:
    st.error(f"⚠️ Could not load data: {e}")
    st.stop()

if df.empty:
    st.info("No violations recorded yet.")
    st.stop()

# Ensure proper types
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Top metrics
st.subheader("🔢 Total Violations")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Helmet", df["no_helmet"].sum())
with col2:
    st.metric("Mask", df["no_mask"].sum())
with col3:
    st.metric("Vest", df["no_vest"].sum())
    
st.markdown("---")
    
col1, col2 = st.columns([1, 1])

with col1:
    latest = df["timestamp"].max()
    st.markdown(f"""
        <h4 style='margin-bottom: 0;'>📅 Latest Violation</h4>
        <h2 style='margin-top: 0;'>{latest.strftime('%Y-%m-%d %H:%M:%S')}</h2>
    """, unsafe_allow_html=True)

with col2:
    most_common = df[["no_helmet", "no_mask", "no_vest"]].sum().idxmax()
    st.markdown(f"""
        <h4 style='margin-bottom: 0;'>🔥 Most Common Violation</h4>
        <h2 style='margin-top: 0;'>{most_common.replace("_", " ").title()}</h2>
    """, unsafe_allow_html=True)


if st.button("🔄 Refresh Data"):
    # Reload your database here
    st.rerun()

st.markdown("---")

col1, col2 = st.columns([1,3])

with col1:
    st.subheader("📊 Violation Type Counts")
    st.bar_chart(df[["no_helmet", "no_mask", "no_vest"]].sum())

with col2:
    st.subheader("📈 Violations Over Time (Daily)")
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")[["no_helmet", "no_mask", "no_vest"]].sum()
    st.bar_chart(daily)#, stack=True)

st.markdown("---")

# Image gallery
st.subheader("📸 Latest Violations")
cols = st.columns(3)
for i, row in df.head(9).iterrows():
    with cols[i % 3]:
        #print(IMAGE_BASE_URL + row["image_path"])
        st.image(IMAGE_BASE_URL + row["image_path"],
                 caption=f"{row['timestamp']}", 
                 use_container_width=True)

st.markdown("---")

# Data table
st.subheader("🗂️ Raw Data")
st.dataframe(df)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Violation Log", csv, "violations.csv", "text/csv")


st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: gray; padding: 1em 0;">
        🚧 PPE Violation Monitoring System – CocaTech Industries © 2025<br>
        Developed by the Computer Vision & Safety AI Lab<br>
        Contact: support@cocatech.ai | Tel: +1 (800) 555-1234
    </div>
""", unsafe_allow_html=True)

