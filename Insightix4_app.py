import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import json
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- PAGE CONFIG ---
st.set_page_config(page_title="Insightix AI Dashboard", layout="wide")

# --- HEADER ---
st.markdown("""
    <h1 style='text-align: center; color: #0072B5; font-size:40px;'>📊 Insightix AI — Interactive Analytics Dashboard</h1>
    <p style='text-align: center; font-size:18px; color:gray;'>Upload your dataset, analyze, visualize, and uncover insights like Power BI ⚡</p>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])

# --- GEMINI API CONFIGURATION ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("⚠️ Gemini API key not found in `.streamlit/secrets.toml`.\n\nAdd this line:\nGEMINI_API_KEY = 'your_api_key_here'")
    st.stop()

# --- IF FILE UPLOADED ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")

    # --- DASHBOARD LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Overview", "📊 Dashboard", "📈 Advanced Analysis", "🧠 AI Insights"])

    # --- TAB 1: DATA OVERVIEW ---
    with tab1:
        st.subheader("📄 Data Preview")
        st.dataframe(df, use_container_width=True, height=400)
        st.markdown("### 📋 Summary Statistics")
        st.write(df.describe(include='all'))

    # --- TAB 2: INTERACTIVE DASHBOARD ---
    with tab2:
        st.subheader("📊 Interactive Power BI-style Dashboard")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if len(numeric_cols) >= 1 and len(cat_cols) >= 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_axis = st.selectbox("Select X-axis", cat_cols)
            with col2:
                y_axis = st.selectbox("Select Y-axis", numeric_cols)
            with col3:
                chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Box", "Pie"])

            if chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, title="Bar Chart")
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color=x_axis, title="Line Chart")
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=x_axis, size=y_axis, title="Scatter Plot")
            elif chart_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis, color=x_axis, title="Box Plot")
            elif chart_type == "Pie":
                fig = px.pie(df, names=x_axis, values=y_axis, title="Pie Chart")

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🧩 Additional Visualizations")
            col4, col5 = st.columns(2)
            with col4:
                st.write("Top 10 by Selected Category")
                top10 = df.groupby(x_axis)[y_axis].sum().nlargest(10).reset_index()
                fig2 = px.bar(top10, x=x_axis, y=y_axis, color=x_axis, title="Top 10 Categories")
                st.plotly_chart(fig2, use_container_width=True)
            with col5:
                st.write("Distribution")
                fig3 = px.histogram(df, x=y_axis, nbins=20, title=f"Distribution of {y_axis}")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("Dataset needs at least one numeric and one categorical column.")

    # --- TAB 3: ADVANCED ANALYSIS ---
    with tab3:
        st.subheader("🧮 Advanced Statistical Analysis")

        analysis_type = st.selectbox("Choose Analysis Type", ["Correlation Matrix", "Regression", "Variance"])
        num_df = df.select_dtypes(include=np.number)

        if analysis_type == "Correlation Matrix":
            if not num_df.empty:
                st.write(num_df.corr())
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric data found.")

        elif analysis_type == "Variance":
            st.write("📊 Variance per Numeric Column:")
            st.write(num_df.var())

        elif analysis_type == "Regression":
            if len(num_df.columns) >= 2:
                x_var = st.selectbox("Select X Variable", num_df.columns)
                y_var = st.selectbox("Select Y Variable", num_df.columns)
                X = df[[x_var]]
                y = df[y_var]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.write(f"**R² Score:** {r2_score(y_test, preds):.3f}")
                st.write(f"**MSE:** {mean_squared_error(y_test, preds):.3f}")
                fig = px.scatter(x=y_test, y=preds, title="Actual vs Predicted",
                                 labels={'x': 'Actual', 'y': 'Predicted'})
                st.plotly_chart(fig)
            else:
                st.warning("Need at least two numeric columns for regression analysis.")

# --- TAB 4: AI INSIGHTS (Gemini Chatbot) ---
else:
    tab4 = st.tabs(["🧠 AI Insights"])[0]

with tab4:
    st.subheader("🤖 Insightix AI Chatbot (Gemini Powered)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("💬 Ask anything (data or general topics)...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Prepare prompt dynamically
        if uploaded_file:
            data_sample = df.head(50).to_dict(orient="records")
            data_json = json.dumps(data_sample)
            prompt = f"""
            You are a professional data analyst. Analyze this dataset:
            {data_json}

            The user asked: "{user_input}"

            Provide a clear, data-driven answer using bullet points, summaries, or calculations where helpful.
            """
        else:
            prompt = f"""
            You are an intelligent assistant. The user asked:
            "{user_input}"

            Provide a thoughtful, clear, and conversational response.
            """

        # Gemini Query
        with st.spinner("💡 Thinking with Gemini..."):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                ai_reply = response.text
            except Exception as e:
                ai_reply = f"⚠️ Gemini error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
