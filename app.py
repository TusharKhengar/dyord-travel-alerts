"""
Streamlit UI for the DYORD traveler news impact analyzer.

Local dev:   LLM_BACKEND=ollama streamlit run app.py
Deployed:    set LLM_BACKEND=groq and GROQ_API_KEY as app secrets, then
             Streamlit Community Cloud / HF Spaces runs `streamlit run app.py`.
"""
import pandas as pd
import streamlit as st

from dyord_pipeline import analyze_location_news

st.set_page_config(page_title="DYORD - Traveler News Impact", page_icon="🧭", layout="centered")

st.title("🧭 DYORD — Traveler News Impact Analyzer")
st.caption(
    "Pulls recent news for a location and flags what's actually relevant to "
    "travelers, with an AI-assessed severity level."
)

with st.form("query"):
    location = st.text_input("Location", placeholder="e.g. Mumbai, Bali, Paris")
    max_articles = st.slider("Articles to scan", min_value=5, max_value=30, value=15)
    submitted = st.form_submit_button("Analyze")

if submitted and location.strip():
    with st.spinner(f"Fetching and analyzing news for {location}..."):
        df = analyze_location_news(location.strip(), max_articles)

    if df.empty:
        st.warning("No articles could be fetched. Try a different location.")
    else:
        relevant = df[df["relevant"]].copy()
        severity_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
        relevant["_rank"] = relevant["severity"].map(severity_order)
        relevant = relevant.sort_values("_rank", ascending=False).drop(columns="_rank")

        st.subheader(f"{len(relevant)} of {len(df)} articles relevant to travelers")

        severity_style = {
            "high": ("🔴", "High"),
            "medium": ("🟠", "Medium"),
            "low": ("🟡", "Low"),
        }

        if relevant.empty:
            st.success("No traveler-relevant concerns found in recent news.")
        for _, row in relevant.iterrows():
            icon, label = severity_style.get(row["severity"], ("⚪", row["severity"]))
            with st.container(border=True):
                st.markdown(f"**{icon} {label} — {row['title']}**")
                st.caption(row["reason"])
                st.markdown(f"[{row['publisher']}]({row['url']}) · {row['published_date']}")

        with st.expander("Show full raw results table"):
            st.dataframe(df, use_container_width=True)

elif submitted:
    st.error("Please enter a location.")
