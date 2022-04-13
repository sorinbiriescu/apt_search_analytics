import streamlit as st
from pg8000.exceptions import DatabaseError
from  pg8000.native import Connection
import datetime as DT
import pandas as pd
import logging
import sklearn.cluster as cluster
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import altair as alt
import math
import pydeck as pdk
from pydeck.types import String
from scipy import stats

platform = st.secrets.get('PLATFORM')
environment = st.secrets.get('APT_SEARCH_ANALYTICS_ENV')
hostname = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_HOSTNAME')
db_user = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_USER')
db_passwd = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_PASSWD')
db_name = st.secrets.get('APT_SEARCH_ANALYTICS_SB_DB_NAME')
log_level = st.secrets.get("LOGLEVEL", "INFO")
mapbox_access_token = st.secrets.get("MAPBOX_ACCESS_TOKEN")

handle = "apt_search_analytics"
logger = logging.getLogger(handle)
logging.basicConfig(level = log_level)

st.set_page_config(page_title="Apt search",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
    )

conn = Connection(db_user,
    host = hostname,
    database = db_name,
    password = db_passwd
    )


@st.cache(suppress_st_warning=True)
def get_last_data_update(request_date = DT.date.today().strftime("%Y-%m-%d")): 
    table_name = environment+".apt_ads_data"

    try:
        result = conn.run(
            f"""SELECT MAX(ad_published_date) FROM {table_name}
            """
            )
        
        return result[0][0].strftime("%d-%m-%Y")

    except DatabaseError as e:
        logging.critical(f"A error occured: {e.args} - {e.message}")
    except Exception as e:
        logging.critical(f"A error occured: {e}") 


@st.cache(suppress_st_warning=True)
def get_unique_locations(request_date = DT.date.today().strftime("%Y-%m-%d")):
    table_name = environment+".apt_ads_data"

    try:
        result = conn.run(
            f"""SELECT DISTINCT apt_location_postal_code
            FROM {table_name}
            """
            )

        logger.debug(result)
    
        return [x[0] for x in result]

    except DatabaseError as e:
        logging.critical(f"A error occured: {e}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")  


@st.cache(suppress_st_warning=True)
def get_data(location = None,
                request_date = DT.date.today().strftime("%Y-%m-%d"),
                start_date = None,
                min_price = None,
                max_price = None,
                min_apt_size = None,
                max_apt_size = None
                ):
 
    table_name = f"{environment}.apt_ads_data"
    table_cols = ["ad_id", "ad_name", "ad_description", "apt_size", "apt_nb_pieces",
        "apt_nb_bedrooms", "apt_location_name", "apt_location_postal_code", "apt_location_lat", "apt_location_long",
        "apt_price", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted",
        "ad_source", "ad_scraping_date"]
    
    query_stmt = f"""SELECT {",".join(table_cols)} FROM {table_name}
        WHERE ad_published_date >= :start_date
            {["AND apt_price >= :min_price" if min_price else ""][0]}
            {["AND apt_price <= :max_price" if max_price else ""][0]}
            {["AND apt_size >= :min_apt_size" if min_apt_size else ""][0]}
            {["AND apt_size <= :max_apt_size" if max_apt_size else ""][0]}
            {["AND apt_location_postal_code = ANY(:location)" if location else ""][0]}
        """
    
    try:
        result = conn.run(query_stmt,
            table_name = table_name,
            start_date = start_date,
            min_price = min_price,
            max_price = max_price,
            min_apt_size = min_apt_size,
            max_apt_size = max_apt_size,
            location = tuple(location)
            )
    
        return result, table_cols

    except DatabaseError as e:
        logging.critical(f"A error occured: {e}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")


def clean_data(df):
    df.drop_duplicates(subset=["ad_name", "apt_location_postal_code", "apt_price"], inplace= True)

    df["ad_id"] = df["ad_id"].astype("string")
    df["ad_name"] = df["ad_name"].astype("string")
    df["ad_description"] = df["ad_description"].astype("string")
    df["apt_location_postal_code"] = df["apt_location_postal_code"].astype("string")

    df["apt_nb_pieces"].fillna(0, inplace = True)
    df["apt_nb_pieces"] = df["apt_nb_pieces"].astype(int)
    df["apt_nb_bedrooms"].fillna(0, inplace = True)
    df["apt_nb_bedrooms"] = df["apt_nb_bedrooms"].astype(int)
    df["apt_price"].fillna(0, inplace = True)
    df["apt_price"] = df["apt_price"].astype(float)
    
    df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d %H:%M:%S", errors = "coerce").dt.strftime("%Y-%m-%d")
    df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d", errors = "coerce")

    df["ad_link"] = df["ad_link"].astype("string")
    df["ad_seller_type"] = df["ad_seller_type"].astype("category")
    df["ad_is_boosted"].fillna("false", inplace = True)
    df["ad_source"] = df["ad_source"].astype("category")

    df["apt_location_lat"] = df["apt_location_lat"].astype(float)
    df["apt_location_long"] = df["apt_location_long"].astype(float)

    return df

def make_link_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'

def add_taxes_to_price(df):
    df["apt_price_w_taxes"] = df["apt_price"]*1.08
    df["apt_price_w_taxes"] = df["apt_price_w_taxes"].astype(int)
    df["apt_price_w_taxes_n_commission"] = df["apt_price"]*1.12
    df["apt_price_w_taxes_n_commission"] = df["apt_price_w_taxes_n_commission"].astype(int)

    return df


def add_ppsqm(df):
    df["apt_price_per_sqm"] = df["apt_price"] / df["apt_size"]
    df["apt_price_per_sqm"].fillna(0, inplace = True)
    df["apt_price_per_sqm"] = df["apt_price_per_sqm"].astype(int)

    return df


def filter_ppsqm(df, filter):
    return df.loc[df["apt_price_per_sqm"].between(filter[0], filter[1], inclusive = "both")]


def deduce_apt_size(df):
    mask_missing_size_price_present = (df["apt_size"].isna() & df["apt_price"].notna() & df["apt_price_per_sqm"].notna())
    df.loc[mask_missing_size_price_present, ("apt_size")] = round(df["apt_price"] /df["apt_price_per_sqm"],2)
    df.loc[df["apt_size"].isnull(), ("apt_size")] = df["ad_name"].str.extract(r'([0-9]{2,3}(?=.*M))', expand = False)
    df["apt_size"] = pd.to_numeric(df["apt_size"], errors = "coerce", downcast = "float")
    df = df.loc[df["apt_size"] < 400]

    return df


def create_price_bins(df):
    df['apt_price_bins'] = pd.cut( x = df['Price'], bins=[0, 50000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 350000, 400000, 500000, 1000000])
    df['apt_price_per_sqm_bins'] = pd.cut( x = df['Price / sqm'], bins=[0, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 10000, 15000, 50000])
    df['apt_size_bins'] = pd.cut( x = df['Size'], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500])

    return df


def run_Kolmogorov_Smirnov_test():
    data = st.session_state["ad_data_global"]

    data['apt_price_per_sqm_bins'] = pd.cut( x = data['Price / sqm'], bins = range(0,12000, 250))
    stats_df_private = data.loc[data["Seller"] == "private", ("ad_id","apt_price_per_sqm_bins")] \
        .groupby("apt_price_per_sqm_bins") \
        .count() \
        .reset_index() \
        .rename(columns = {'ad_id': 'frequency'}) \
        .sort_values(by = "apt_price_per_sqm_bins", ascending = True)
    stats_df_private['pdf'] = stats_df_private['frequency'] / sum(stats_df_private['frequency'])
    stats_df_private['cdf'] = stats_df_private['pdf'].cumsum()
    stats_df_private["apt_price_per_sqm_bins"] = stats_df_private["apt_price_per_sqm_bins"].astype("string")
    
    stats_df_pro = data.loc[data["Seller"] == "pro", ("ad_id","apt_price_per_sqm_bins")] \
        .groupby("apt_price_per_sqm_bins") \
        .count() \
        .reset_index() \
        .rename(columns = {'ad_id': 'frequency'}) \
        .sort_values(by = "apt_price_per_sqm_bins", ascending = True)
    stats_df_pro['pdf'] = stats_df_pro['frequency'] / sum(stats_df_pro['frequency'])
    stats_df_pro['cdf'] = stats_df_pro['pdf'].cumsum()
    stats_df_pro["apt_price_per_sqm_bins"] = stats_df_pro["apt_price_per_sqm_bins"].astype("string")


    statistic, p_value = stats.ks_2samp(stats_df_private["cdf"], stats_df_pro["cdf"], alternative = "two-sided")
    
    st.markdown("The null hypothesis is that the price / sqm distribution of private sellers is \
        the same as pro ones")
    st.write(f"statistic: `{statistic}`")
    st.write(f"p value: `{p_value}`")

    alpha_level = 0.05

    if p_value <= alpha_level:
        st.markdown("Null hypothesis rejected - Price distributions different")
    else:
        st.markdown("Failed to reject the null hypothesis - Price distributions are the same")


    chart_pdf_private = alt.Chart(stats_df_private).mark_line(interpolate='step-after').encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_private["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False)),
            y='pdf'
        )

    chart_pdf_pro = alt.Chart(stats_df_pro).mark_line(interpolate='step-after', color="#FFAA00").encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_pro["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False)),
            y='pdf'
        )

    chart_cdf_private = alt.Chart(stats_df_private).mark_line(interpolate='step-after').encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_private["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False)),
            y='cdf'
        )

    chart_cdf_pro = alt.Chart(stats_df_pro).mark_line(interpolate='step-after', color="#FFAA00").encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_pro["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False)),
            y='cdf'
        )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_pdf_private + chart_pdf_pro, use_container_width= True)
    with col2:
        st.altair_chart(chart_cdf_private + chart_cdf_pro, use_container_width= True)
        

    return


def generate_price_distributions():
    data = st.session_state["ad_data_global"]
    data['apt_size_bins'] = pd.cut( x = data['Size'], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500])
    data = data.loc[:, ("apt_size_bins", "Price / sqm", "Seller")] \
        .groupby(["apt_size_bins", "Seller"]) \
        .mean() \
        .reset_index() \
        .sort_values(by = "apt_size_bins", ascending = True)

    data["apt_size_bins"] = data["apt_size_bins"].astype("string")
    data["Seller"] = data["Seller"].astype("string")

    # chart = alt.Chart(data).mark_bar().encode(
    #     x = alt.X('Seller:O'),
    #     y = 'Price / sqm:Q',
    #     color = 'Seller:N',
    #     column = alt.Column('apt_size_bins:N', 
    #         sort = data["apt_size_bins"].tolist()
    #         )
    #     )

    chart = alt.Chart(data).mark_bar().encode(
        x = alt.X('Seller:O'),
        y = 'Price / sqm:Q',
        color = 'Seller:N',
        facet = alt.Facet('apt_size_bins:N', 
            columns= 4,
            sort = data["apt_size_bins"].tolist()
            )
        )

    st.altair_chart(chart)

    return


def generate_cluster_chart(df, k, show_elbow = False):
    if len(df) <= 15:
        return None, None

    mask_no_apt_size = df["Size"].isna()
    mask_no_price_per_sqm = df["Price / sqm"].isna()
    mask_no_postal_code = df["Location"].isna()
    df_null_values = df.loc[mask_no_apt_size | mask_no_price_per_sqm | mask_no_postal_code]
    df = df.loc[~df.index.isin(df_null_values.index)]

    apt_size_scaler = StandardScaler()
    df["apt_size_norm"] = apt_size_scaler.fit_transform(df[["Size"]])

    apt_price_per_sqm_scaler = StandardScaler()
    df["apt_price_per_sqm_norm"] = apt_price_per_sqm_scaler.fit_transform(df[["Price / sqm"]])

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit_transform(df[["Location"]])

    kmeans = cluster.KMeans(n_clusters = k ,init = "k-means++")
    kmeans.fit(df[["apt_price_per_sqm_norm", "apt_size_norm"]])
    df['clusters'] = kmeans.labels_

    cluster_chart = alt.Chart(df.loc[:,("Size","Price / sqm","clusters")]).mark_circle().encode(
        x = "Size:Q",
        y = "Price / sqm:Q",
        color = "clusters:N"
    ).properties(
        title = 'Price per sqm per size',
        width = 800,
        height = 400
    ).interactive()

    if show_elbow:
        # generate elbow
        nb_clusters = list(range(1, 15))
        sse = list()
        for k in nb_clusters:
            kmeans_tester = cluster.KMeans(n_clusters = k ,init = "k-means++")
            kmeans_tester.fit(df[["apt_price_per_sqm_norm", "apt_size_norm"]])
            sse.append({"clusters": k, "inertia": kmeans_tester.inertia_})

        elbow_chart = alt.Chart(pd.DataFrame(sse)).mark_line().encode(
            x = "clusters:O",
            y = "inertia:Q"
        )
        
        return cluster_chart, elbow_chart

    else:
        return cluster_chart, None


def generate_results_on_map(data):
    data.dropna(axis = 0, subset = ["apt_location_long", "apt_location_lat"], inplace = True)
    data["index"] = data.index + 1
    data["index"] = data["index"].astype(str)

    layer = pdk.Layer(
        'ScatterplotLayer',
        data,
        get_position = ["apt_location_long", "apt_location_lat"],
        pickable=True,
        stroked=True,
        filled=True,
        radius_min_pixels = 5,
        radius_max_pixels = 40,
        line_width_min_pixels = 1,
        get_fill_color = [255, 140, 0]
        )

    layer_text = pdk.Layer(
        "TextLayer",
        data,
        pickable = False,
        get_position = ['apt_location_long', 'apt_location_lat'],
        get_text = "index",
        radius_min_pixels = 5,
        radius_max_pixels = 40,
        get_color = [255, 140, 0],
        get_angle = 0,
        # Note that string constants in pydeck are explicitly passed as strings
        # This distinguishes them from columns in a data set
        get_text_anchor=String("middle"),
        get_alignment_baseline=String("bottom")
        )

    view_state = pdk.ViewState(
        longitude = 4.834277,
        latitude = 45.763420,
        zoom = 12,
        min_zoom=5,
        max_zoom=15
        )

    return pdk.Deck(layers = [layer_text, layer],
        initial_view_state = view_state,
        tooltip={"text": "{index}"},)


def create_distr_chart(data, bin_name, group_name):
    chart_data = data.loc[:,("ad_id", bin_name)].groupby(bin_name).count().reset_index()
    chart_data.sort_values(by = [bin_name], inplace = True, ascending = True)
    chart_data[bin_name] = chart_data[bin_name].astype(str)

    chart = alt.Chart(chart_data).mark_bar().encode(
        x = alt.X(f"{bin_name}:O", sort = chart_data[bin_name].tolist(),axis=alt.Axis(title= group_name)),
        y = alt.Y("ad_id",  axis=alt.Axis(title= "Total count"))
    )

    return chart


def generate_market_analysis(scope = "local"):
    st.number_input("# of groups - local",
        key = f"nb_clusters_{scope}",
        min_value = 0,
        max_value = 15,
        format = "%i",
        value = 5)
    st.checkbox('Show cluster elbow chart',
        key= f"show_elbow_chart_{scope}", value = False)

    k = st.session_state[f"nb_clusters_{scope}"]
    data = st.session_state[f"ad_data_{scope}"]

    data = create_price_bins(data)


    cluster_chart, elbow_chart = generate_cluster_chart(data, k ,show_elbow= st.session_state[f"show_elbow_chart_{scope}"])


    if cluster_chart:
        st.altair_chart(cluster_chart, use_container_width=True)
    if elbow_chart:
        st.altair_chart(elbow_chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_distr_chart(data, "apt_price_bins", "Total price"))
    with col2:
        st.altair_chart(create_distr_chart(data, "apt_price_per_sqm_bins", "Price / sqm"))

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_distr_chart(data, "apt_size_bins", "Apartment size"))


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

        st.markdown(f'Price per sqm: `{row["Price / sqm"]:,}`')

        st.markdown(row["Link"])

        r_col7, r_col8, r_col9 = st.columns([1,1,2])
        with r_col9:
            st.markdown(f'Ad publish date: **{row["Published date"]}**')
        with r_col8:
            st.markdown(f'Seller type: **{row["Seller"]}**')
        with r_col7:
            st.markdown(f'Boosted ad: **{row["Boosted"]}**')

        st.markdown(f"***")

def subset_results(df):
    DATATABLE_COLS = ["ad_name", "ad_description", "apt_size", "apt_nb_pieces",
    "apt_location", "apt_price", "apt_price_w_taxes","apt_price_w_taxes_n_commission", 
    "apt_price_per_sqm", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted",
    "apt_location_lat", "apt_location_long"]

    return df.loc[:, DATATABLE_COLS]


def sort_values(df):
    return df.sort_values(by = st.session_state.get("sort_option", "apt_price_per_sqm"),
        ascending = st.session_state.get("sort_direction", True))


def rename_columns(df):
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
        "ad_is_boosted": "Boosted"}

    return df.rename(columns = RENAME_MAPPING)


def reset_index(df):
    return df.reset_index(drop = True)


def run_data_pipeline(scope = "local"):
    time_hor = [st.session_state["start_date"] if scope == "local" else DT.datetime.now() - DT.timedelta(days = 365)][0]

    data, data_cols = get_data(start_date = time_hor,
        min_price = int(st.session_state["price"][0]),
        max_price = int(st.session_state["price"][1]),
        min_apt_size = int(st.session_state["apt_size"][0]),
        max_apt_size = int(st.session_state["apt_size"][1]),
        location = st.session_state["locations"]
        )

    st.session_state[f"ad_data_{scope}"] = pd.DataFrame(data, columns = data_cols) \
        .pipe(clean_data) \
        .pipe(add_taxes_to_price) \
        .pipe(add_ppsqm) \
        .pipe(filter_ppsqm, st.session_state["ppsqm"]) \
        .pipe(deduce_apt_size) \
        .pipe(sort_values) \
        .pipe(rename_columns) \
        .pipe(reset_index)
    
    return

SORT_OPTIONS = {"apt_price_per_sqm": "Price per sqm",
        "apt_price":"Apt. price",
        "apt_price_w_taxes": "Apt. price with taxes",
        "apt_size": "Apartment size"}

SORT_DIRECTION = {True: "Ascending",
    False: "Descending"}

def format_sort_options(option):
    return SORT_OPTIONS.get(option)

def format_sort_direction(option):
    return SORT_DIRECTION.get(option)


# RESULTS PAGE ------------------------------------------------------------------

# Sidebar form
with st.form(key = "filter_criteria"):
    with st.sidebar:
        st.markdown('# Filters')
        st.date_input(
            "Ads since publish date",
            DT.datetime.now() - DT.timedelta(days = 3),
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

        location_list = get_unique_locations()
        location_multiselect = st.multiselect(
                        label = "Locations",
                        options = location_list,
                        default = location_list,
                        key = "locations"
                        )

        submit = st.form_submit_button(label="Submit", help=None)   

run_data_pipeline()

# Main page

st.markdown('# Apartment search Lyon')
st.write('👈 Please use the form in the sidebar to filter results')
st.markdown(f"Last data update: `{get_last_data_update()}`")
total_results = len(st.session_state["ad_data_local"])
st.markdown(f'Total results: `{total_results}`')
st.markdown(f"***")

st.checkbox('Show results', key= "show_filtered_results", value = True)
if st.session_state["show_filtered_results"]:
    total_results = len(st.session_state["ad_data_local"])
    results_per_page = 5
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

        st.markdown(f'Showing page **{st.session_state["display_results_page_number"]+1}** of total **{total_pages+1}** pages')
        
        st.pydeck_chart(generate_results_on_map(page_result))
        st.markdown(f"***")
        
        for index, row in page_result.iterrows():
            generate_result_entry(index, row)

        st.markdown(f'Showing page **{st.session_state["display_results_page_number"]+1}** of total **{total_pages+1}** pages')

st.markdown("***")
st.checkbox('Show market analysis on filtered results', key= "show_local_analysis", value = False)
if st.session_state["show_local_analysis"]:
    st.markdown('## Market analysis')
    st.markdown('### Filtered data analysis')
    generate_market_analysis()

st.markdown("***")
st.checkbox('Show global analysis', key= "show_global_analysis", value = False)
if st.session_state["show_global_analysis"]:
    st.markdown('## Global data analysis')
    generate_market_analysis(scope = "global")

    st.markdown('### Kolmogorov Smirnov test')
    run_Kolmogorov_Smirnov_test()

    st.markdown('### Price distributions')
    generate_price_distributions()