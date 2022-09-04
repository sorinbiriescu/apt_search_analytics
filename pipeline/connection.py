from  pg8000.native import Connection
import streamlit as st

hostname = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_HOSTNAME')
db_user = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_USER')
db_passwd = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_PASSWD')
db_name = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_NAME')

conn = Connection(db_user,
    host = hostname,
    database = db_name,
    password = db_passwd
    )
