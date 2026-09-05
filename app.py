"""
Streamlit UI for the DYORD traveler news impact analyzer.

Local dev:   LLM_BACKEND=ollama streamlit run app.py
Deployed:    set LLM_BACKEND=groq and GROQ_API_KEY as app secrets, then
             Streamlit Community Cloud / HF Spaces runs `streamlit run app.py`.
"""
import pandas as pd
import streamlit as st

from dyord_pipeline import analyze_location_news
from llm_backend import LLM_BACKEND

st.set_page_config(page_title="DYORD - Traveler News Impact", page_icon="🧭", layout="centered")

st.title("🧭 DYORD — Traveler News Impact Analyzer")
st.caption(
    "Pulls recent news for a location and flags what's actually relevant to "
    "travelers, with an AI-assessed severity level."
)

# Debug info - shows which backend is active
with st.expander("🔧 Debug Info"):
    st.text(f"LLM Backend: {LLM_BACKEND}")

    if LLM_BACKEND == "groq":
        from llm_backend import _get_secret
        api_key = _get_secret("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                models = client.models.list()
                st.text(f"✅ API Key: Valid (starts with {api_key[:10]}...)")
                st.text(f"Available models: {len(models.data)}")
                if models.data:
                    st.text("Models you can use:")
                    for m in models.data[:5]:
                        st.text(f"  - {m.id}")
                else:
                    st.warning("⚠️ No models available with this API key!")
            except Exception as e:
                st.error(f"❌ API Key Error: {e}")
        else:
            st.error("❌ GROQ_API_KEY not found in secrets")

with st.form("query"):
    location = st.text_input("Location", placeholder="e.g. Mumbai, Bali, Paris")
    max_articles = st.slider("Articles to scan", min_value=5, max_value=30, value=15)
    submitted = st.form_submit_button("Analyze")

if submitted and location.strip():
    with st.spinner(f"Fetching and analyzing news for {location}..."):
        try:
            df = analyze_location_news(location.strip(), max_articles)
        except Exception as e:
            st.error(f"❌ Analysis failed: {type(e).__name__}: {e}")
            st.stop()

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

        # Check if classifications actually succeeded or failed with errors
        failed_rows = df[df["reason"].str.startswith("classification failed:", na=False)]
        if len(failed_rows) == len(df) and len(df) > 0:
            first_error = failed_rows.iloc[0]["reason"]
            st.error(f"❌ Groq API Error: {first_error}")
            st.info("💡 Please verify that your `GROQ_API_KEY` in Streamlit Secrets is active and valid.")
        elif relevant.empty:
            st.success("No traveler-relevant concerns found in recent news.")
        else:
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
