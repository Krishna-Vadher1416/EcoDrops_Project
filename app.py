import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# Set Streamlit page config
st.set_page_config(page_title="EcoDrops Dashboard", layout="wide")

# Load your data
df = pd.read_csv('data/water_usage_with_prediction.csv')


# Title and description
st.title("🌿 EcoDrops: AI-Powered Water Conservation")
# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🌍 Trends", "🚨 Anomalies", "🧠 AI Tips"])

st.markdown("Visualize and track sustainable water usage using AI insights.")
with tab1:
    # Sidebar filters
    country = st.sidebar.selectbox("Select Country", sorted(df['Country'].unique()))
    year = st.sidebar.selectbox("Select Year", sorted(df['Year'].unique()))

    # Filter data based on selection
    filtered = df[(df['Country'] == country) & (df['Year'] == year)]

    # Show selected data
    if not filtered.empty:
        row = filtered.iloc[0]
            # Gauge chart for Water Score
        st.markdown("### 💧 Water Efficiency Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=row['Water Score'],
            title={'text': "Water Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#F1948A"},    # Red
                    {'range': [40, 70], 'color': "#F9E79F"},   # Yellow
                    {'range': [70, 100], 'color': "#82E0AA"}   # Green
                ]
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=False, width=400, height=300)


        st.subheader(f"📍 {row['Country']} — {int(row['Year'])}")


        st.metric("💧 Water Efficiency Score", f"{int(row['Water Score'])}/100")
        st.markdown("### 🤖 AI-Powered Prediction (Deep Learning)")

        col1, col2 = st.columns(2)
        col1.metric("📊 Actual Water Score", f"{int(row['Water Score'])}/100")
        col2.metric("🧠 Predicted Score (DL)", f"{int(row['Predicted Score'])}/100")

        diff = abs(row['Water Score'] - row['Predicted Score'])

        if diff <= 5:
            st.success("✅ Model prediction is very accurate!")
        elif diff <= 15:
            st.warning("⚠️ Acceptable prediction — some deviation.")
        else:
            st.error("❌ Model needs tuning — prediction is quite different.")


        st.markdown(f"### 🌱 Sustainability Tip:")
        st.info(row['Sustainability Tip'])

        # Water Use Distribution Chart
        
        fig, ax = plt.subplots(figsize=(6,4))
        usage_labels = ['Agricultural', 'Industrial', 'Household']
        usage_values = [row['Agricultural Water Use (%)'], row['Industrial Water Use (%)'], row['Household Water Use (%)']]
        sns.barplot(x=usage_labels, y=usage_values, palette="coolwarm", ax=ax)
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 100)
        
        bar_fig= fig
    else:
        st.warning("No data available for selected filters.")
    

    fig3, ax3 = plt.subplots()
    labels = ['Agricultural', 'Industrial', 'Household']
    sizes = [row['Agricultural Water Use (%)'], row['Industrial Water Use (%)'], row['Household Water Use (%)']]
    colors = ['#A3E4D7', '#F9E79F', '#F5B7B1']
    explode = (0.05, 0.05, 0.05)  # explode all slices
    
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
    ax3.axis('equal')
    
    pie_fig = fig3

    

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Water Use Distribution (%)")
        st.pyplot(bar_fig)

    with col2:
        st.markdown("### 🥧 Water Use Breakdown (Pie Chart)")
        st.pyplot(pie_fig)
        
    # Download button for filtered row
    st.download_button(
        label="⬇️ Download This Report as CSV",
        data=filtered.to_csv(index=False),
        file_name=f"{row['Country']}_{int(row['Year'])}_water_report.csv",
        mime='text/csv'
    )
    st.markdown("## 🧠 AI-Driven Recommendations")
    if row['Water Score'] < 50:
        st.error("🔴 High Water Stress! Implement stricter water-saving measures.")
    elif row['Water Score'] < 80:
        st.warning("🟠 Medium Risk: Consider awareness campaigns.")
    else:
        st.success("🟢 Sustainable water use detected. Keep monitoring!")

with tab2:
    
    st.markdown("## 📉 Water Score Trend Over Time")

    country_data = df[df['Country'] == country]

    fig2, ax2 = plt.subplots(figsize=(8, 4))  # Adjusted size if needed

    sns.lineplot(data=country_data, x='Year', y='Water Score', marker='o', ax=ax2, color='dodgerblue')

    ax2.set_title(f"Water Score Trend for {country}")
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Water Score")
    

    # ➕ Add labels above each point
    for x, y in zip(country_data['Year'], country_data['Water Score']):
        ax2.text(x, y + 1.5, f"{y:.1f}", ha='center', fontsize=8, color='black')

    st.pyplot(fig2)


with tab3:
    st.markdown("## 🚨 Global Water Overuse Alerts (Anomalies Detected)")
    anomalies = df[df['Anomaly'] == -1][['Country', 'Year', 'Per Capita Water Use (Liters per Day)', 'Groundwater Depletion Rate (%)', 'Water Score']]
    st.dataframe(anomalies.reset_index(drop=True), use_container_width=True)

with tab4:
    st.markdown("#### 🌍 Empowering Climate Action Through AI 🌿")


    st.markdown("## 🧠 How AI & ML Support Sustainability")

    st.info("AI can detect abnormal water usage patterns using models like Isolation Forest. This helps identify regions with high wastage or stress.")
    st.markdown("---")

    st.success("Smart irrigation systems powered by AI can optimize water usage based on weather predictions and soil moisture.")
    st.markdown("---")

    st.warning("Predictive models can forecast water demand, allowing early policy action to prevent shortages.")
    st.markdown("---")

    st.info("AI-powered dashboards (like this one!) make climate data easy to understand for policymakers and citizens.")
    st.markdown("---")

    st.success("ML can help design smart infrastructure and support green city planning by analyzing environmental datasets.")
    st.markdown("---")
    with st.expander("🧠 How does the AI Model Work?"):
        st.markdown("""
        Our model uses a **Feedforward Neural Network (FNN)** trained on real-world water usage data.  
        It analyzes five key inputs:
        - 💧 Per Capita Water Use
        - 🌊 Groundwater Depletion Rate
        - 🌾 Agricultural Water Use
        - 🏭 Industrial Water Use
        - 🏠 Household Water Use

        Based on this, it predicts the **Water Efficiency Score** for a given region and year.  
        This score helps governments and planners:
        - Create sustainable water policies
        - Monitor regions at risk
        - Promote green infrastructure and awareness

        ✅ Built with TensorFlow and Streamlit.
        """)
