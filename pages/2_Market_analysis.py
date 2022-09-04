import streamlit as st
import datetime as DT
import pandas as pd

from pipeline.connection import conn
from pipeline.collection import get_last_data_update, get_unique_locations, \
    get_ad_data, get_dvf_data
from pipeline.cleaning import clean_ad_data, make_link_clickable
from pipeline.transformation import add_taxes_to_price, add_ppsqm, \
    filter_ppsqm, deduce_apt_size, create_price_bins, align_dvf_column_names, \
    add_record_type, calculate_price_delta, merge_ad_dvf_data, sort_values, \
        reset_index
from pipeline.analysis import run_Kolmogorov_Smirnov_test, generate_market_analysis
from pipeline.visualization import generate_cluster_chart, create_distr_chart, \
    generate_results_on_map, generate_price_distributions, generate_price_evolution


def run_ad_data_pipeline_global_analysis():
    ad_table_cols = ["ad_id", "ad_name","apt_size", "apt_nb_pieces", "apt_location_postal_code",
        "apt_price", "ad_published_date", "ad_seller_type", "ad_is_boosted"]

    ad_data, result_cols = get_ad_data(table_cols= ad_table_cols,
        start_date = DT.datetime.now().date() - DT.timedelta(days = 365)
        )

    dvf_table_cols = ["date_of_transaction", "sell_value", "postal_code", "real_build_surface",
        "carrez_surface", "rooms_number"]

    dvf_data, dvf_data_cols = get_dvf_data(table_cols = dvf_table_cols,
        start_date = DT.datetime.now().date() - DT.timedelta(days = 365)
        )

    dvf_df = pd.DataFrame(dvf_data, columns = dvf_data_cols) \
        .pipe(align_dvf_column_names) \
        .pipe(add_record_type, "dvf")

    st.session_state[f"ad_data_global"] = pd.DataFrame(ad_data, columns = result_cols) \
        .pipe(calculate_price_delta) \
        .pipe(add_record_type, "ads") \
        .pipe(merge_ad_dvf_data, dvf_df = dvf_df) \
        .pipe(clean_ad_data) \
        .pipe(add_taxes_to_price) \
        .pipe(add_ppsqm) \
        .pipe(deduce_apt_size) \
        .pipe(sort_values) \
        .pipe(reset_index)
    
    return

st.markdown('## Global data analysis')
st.markdown("The global data analysis is run on a time horizon of 365 days, \
    regardless of the filters selected locally.")
run_ad_data_pipeline_global_analysis()
generate_market_analysis(scope = "global")

st.markdown('### Kolmogorov Smirnov test')
run_Kolmogorov_Smirnov_test()

st.markdown('### Price distributions')
st.altair_chart(generate_price_distributions())

st.markdown('### Price evolution')
st.markdown("""As the price is very different from zone to zone (ex: zones closer to the \
edge of the city compared to posh areas, the price evolution is best to be presented \
per apartment size class and postal code). \

Prices do not contain ads from developers, brokers and networks, as they seem overall \
higher than the 'old' buildings."""
    )
generate_price_evolution(time_sample = "weekly")