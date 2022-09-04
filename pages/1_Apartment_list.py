import streamlit as st
import datetime as DT
import pandas as pd
import logging
import math


from Main import logger
from pipeline.collection import get_last_data_update, get_unique_locations, \
    get_ad_data, get_dvf_data
from pipeline.cleaning import clean_ad_data, make_link_clickable
from pipeline.transformation import add_taxes_to_price, add_ppsqm, \
    filter_ppsqm, deduce_apt_size, create_price_bins, align_dvf_column_names, \
    calculate_price_delta, sort_values, reset_index, transform_to_cat
from pipeline.analysis import run_Kolmogorov_Smirnov_test, generate_market_analysis
from pipeline.visualization import generate_cluster_chart, create_distr_chart, \
    generate_results_on_map
from pipeline.prediction import predict_prices

def _debug_generate_time(df, step):
    logging.info(f"Step {step}: time {DT.datetime.now()}")
    return df


def generate_result_entry(index, row):
        st.markdown(f"### {index + 1} - {row['Name']}")
        st.markdown(f"Description: {row['Description'][:300]} ...")

        r_col1, r_col2, r_col3 = st.columns([1,1,2])
        with r_col3:
            st.markdown(f'Location: `{row["Location"]}`')
        with r_col2:
            st.markdown(f'Rooms: `{row["Rooms"]:.0f}`')
        with r_col1:
            st.markdown(f'Size: `{row["Size"]:.0f}`')


        r_col4, r_col5, r_col6 = st.columns([1,1,2])
        with r_col4:
            st.markdown(f'Price: `{row["Price"]:,}`')
        with r_col5:
            st.markdown(f'Price w: taxes: `{row["Price w. taxes"]:,}`')
        with r_col6:
            st.markdown(f'Price w. taxes & comm: `{row["Price w.taxes & comm"]:,}`')

        r_col13, r_col14, r_col15 = st.columns([1,1,2])
        with r_col13:
            st.markdown(f'Predicted: `{row["predicted_price"]:,}`')

        r_col7, r_col8, r_col9 = st.columns([1,1,2])
        with r_col7:
            st.markdown(f'Price per sqm: `{row["Price / sqm"]:,}`')
        with r_col8:
            st.markdown(f'First price: `{row["first_price"]:,}`')
        with r_col9:
            st.markdown(f'Price difference: `{row["price_delta"]:,}`')

        st.markdown(row["Link"])

        r_col10, r_col11, r_col12 = st.columns([1,1,2])
        with r_col12:
            st.markdown(f'Ad publish date: **{row["Published date"]}**')
        with r_col11:
            st.markdown(f'Seller type: **{row["Seller"]}**')
        with r_col10:
            st.markdown(f'Boosted ad: **{row["Boosted"]}**')

        st.markdown(f"***")

# def subset_results(df):
#     DATATABLE_COLS = ["ad_name", "ad_description", "apt_size", "apt_nb_pieces",
#     "apt_location", "apt_price", "apt_price_w_taxes","apt_price_w_taxes_n_commission", 
#     "apt_price_per_sqm", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted",
#     "apt_location_lat", "apt_location_long"]

#     return df.loc[:, DATATABLE_COLS]

# def add_dvf_data(df):
#     return df


def run_ad_data_pipeline_local_results():
    ad_table_cols = ["ad_id", "ad_name", "ad_description", "apt_size", "apt_nb_pieces",
        "apt_nb_bedrooms", "apt_location_name", "apt_location_postal_code", "apt_location_lat", "apt_location_long",
        "apt_price", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted",
        "ad_source", "ad_scraping_date"]

    data, result_cols = get_ad_data(table_cols= ad_table_cols,
        start_date = st.session_state["start_date"],
        min_price = int(st.session_state["price"][0]),
        max_price = int(st.session_state["price"][1]),
        min_apt_size = int(st.session_state["apt_size"][0]),
        max_apt_size = int(st.session_state["apt_size"][1]),
        min_nb_rooms = int(st.session_state["nb_rooms"][0]),
        max_nb_rooms = int(st.session_state["nb_rooms"][1]),
        location = st.session_state["locations"]
        )

    st.session_state[f"ad_data_local"] = pd.DataFrame(data, columns = result_cols) \
        .pipe(clean_ad_data) \
        .pipe(transform_to_cat, "apt_location_postal_code") \
        .pipe(calculate_price_delta) \
        .pipe(add_taxes_to_price) \
        .pipe(add_ppsqm) \
        .pipe(filter_ppsqm, st.session_state["ppsqm"]) \
        .pipe(deduce_apt_size) \
        .pipe(predict_prices) \
        .pipe(sort_values) \
        .pipe(reset_index)

        #.pipe(rename_columns) \
    
    return


SORT_OPTIONS = {"apt_price_per_sqm": "Price per sqm",
        "apt_price":"Apt. price",
        "apt_price_w_taxes": "Apt. price with taxes",
        "apt_size": "Apartment size",
        "ad_published_date": "Date posted",
        "price_delta": "Price difference"}

SORT_DIRECTION = {True: "Ascending",
    False: "Descending"}

def format_sort_options(option):
    return SORT_OPTIONS.get(option)

def format_sort_direction(option):
    return SORT_DIRECTION.get(option)


# RESULTS PAGE ------------------------------------------------------------------

# Sidebar form
#------------------------------------------------------------------
with st.form(key = "filter_criteria"):
    with st.sidebar:
        st.markdown('# Filters')
        st.date_input(
            "Active ads since date",
            DT.datetime.now() - DT.timedelta(days = 7),
            key = "start_date")
        start_price_slider, end_price_slider = st.select_slider('Apartment Price',
            options=[price for price in range(0,525000,25000)],
            key = "price",
            value = [0, 500000]
        )

        start_ppsqm_slider, end_ppsqm_slider = st.select_slider('Price per square meter',
            options=[price for price in range(0,25500,500)],
            key = "ppsqm",
            value = [0, 25000]
            )

        start_apt_size_slider, end_apt_size_slider = st.select_slider('Apartment size',
            options=[size for size in range(0, 305, 5)],
            key = "apt_size",
            value = [0, 300]
            )

        start_nb_rooms_slider, end_nb_rooms_slider = st.select_slider('Number of rooms',
            options=[size for size in range(0, 11, 1)],
            key = "nb_rooms",
            value = [0, 10]
            )

        location_list = get_unique_locations()
        location_multiselect = st.multiselect(
                        label = "Locations",
                        options = location_list,
                        default = location_list,
                        key = "locations"
                        )

        submit = st.form_submit_button(label="Submit", help=None)   

#------------------------------------------------------------------

run_ad_data_pipeline_local_results()

# Main page

st.markdown('# Apartment search Lyon')
st.write('👈 Please use the form in the sidebar to filter results')
total_results = len(st.session_state["ad_data_local"])
results_per_page = 5
total_results = len(st.session_state["ad_data_local"])
total_pages = math.ceil(total_results / results_per_page)

results_container = st.empty()

col1, col2, col3 = st.columns([1,10,1])

if "display_results_page_number" not in st.session_state:
    st.session_state['display_results_page_number'] = 0

with col1:
    if st.button("Previous"):
        st.session_state["display_results_page_number"] -= 1

with col3:
    if st.button("Next"):
        st.session_state["display_results_page_number"] += 1

start = st.session_state["display_results_page_number"] * results_per_page
end = start + results_per_page - 1
page_result = st.session_state["ad_data_local"].loc[start:end]


with results_container.container():
    with st.form(key= "sort_options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Sort by:",
                (SORT_OPTIONS.keys()),
                format_func= format_sort_options,
                key= "sort_option")
        with col2:
            st.selectbox("Direction",
                (SORT_DIRECTION.keys()),
                format_func= format_sort_direction,
                key= "sort_direction")
        with col3:
            st.form_submit_button(label= "Apply")

    st.markdown(f'Total results: **{total_results}** - Showing page **{st.session_state["display_results_page_number"]+1}** of total **{total_pages}** pages')
    
    try:
        st.pydeck_chart(generate_results_on_map(page_result))
    except:
        pass

    st.markdown("***")
    st.checkbox('Show market analysis on filtered results', key= "show_local_analysis", value = False)
    if st.session_state["show_local_analysis"]:
        st.markdown('## Market analysis')
        st.markdown('### Filtered data analysis')
        generate_market_analysis()

    st.markdown(f"***")

    RENAME_MAPPING = {"ad_name": "Name",
        "ad_description": "Description",
        "apt_size": "Size",
        "apt_nb_pieces": "Rooms",
        "apt_location_name": "Location",
        "apt_price": "Price",
        "apt_price_w_taxes": "Price w. taxes",
        "apt_price_w_taxes_n_commission": "Price w.taxes & comm",
        "apt_price_per_sqm": "Price / sqm",
        "ad_link": "Link",
        "ad_published_date": "Published date",
        "ad_seller_type": "Seller",
        "ad_is_boosted": "Boosted"
        }

    page_result.rename(columns = RENAME_MAPPING, inplace = True)
    
    for index, row in page_result.iterrows():
        generate_result_entry(index, row)

    st.markdown(f'Total results: **{total_results}** - Showing page **{st.session_state["display_results_page_number"]+1}** of total **{total_pages}** pages')


