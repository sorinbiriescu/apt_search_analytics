import streamlit as st
import logging
from pipeline.collection import get_last_data_update, get_ads_total, \
    get_price_points_total, get_sold_entries_total

log_level = st.secrets.get("LOGLEVEL", "INFO")

handle = "apt_search_analytics"
logger = logging.getLogger(handle)
logging.basicConfig(level = log_level)

platform = st.secrets.get('PLATFORM')
environment = st.secrets.get('APT_SEARCH_ANALYTICS_ENV')
mapbox_access_token = st.secrets.get("MAPBOX_ACCESS_TOKEN")


st.set_page_config(page_title="Apt search",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
    )

# Main page
st.markdown('# Apartment search Lyon')
st.markdown(f'Last update: `{get_last_data_update()}`')
st.markdown(f'## Volume of data')
st.markdown(f'### Total ads: {get_ads_total()}')
st.markdown(f'Total ads collected from websites. Contains duplicates as often one ad is posted on \
    several websites')
st.markdown(f'### Total price points: {get_price_points_total()}')
st.markdown(f'Total price collected per active ad each day. This is to track price evolution \
    per ad')
st.markdown(f'### Total sold entries: {get_sold_entries_total()}')
st.markdown(f'Data collected from DVF. It\'s used to compare selling price to ad price')