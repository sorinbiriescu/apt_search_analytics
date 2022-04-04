import streamlit as st
from pg8000.exceptions import DatabaseError
from  pg8000.native import Connection
import datetime as DT
import pandas as pd
import logging
import plotly.graph_objects as go
import sklearn.cluster as cluster
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import altair as alt


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

logger.debug("Current environment: {}".format(environment))
logger.debug("DB hostname: {}".format(hostname))
logger.debug("DB passw: {}".format(db_passwd))


conn = Connection(db_user,
    host = hostname,
    database = db_name,
    password = db_passwd
    )


@st.cache(suppress_st_warning=True)
def get_last_data_update(request_date = DT.date.today().strftime("%Y-%m-%d")):
    logger.debug("Cache miss, get_last_data_update function ran")
    logger.debug(f"Request date: {request_date}")
    
    table_name = environment+".apt_ads_data"

    try:
        result = conn.run(
            f"""SELECT MAX(ad_published_date) FROM {table_name}
            """
            )
        
        return result

    except DatabaseError as e:
        logging.critical(f"A error occured: {e.args} - {e.message}")
    except Exception as e:
        logging.critical(f"A error occured: {e}") 


@st.cache(suppress_st_warning=True)
def get_unique_locations(request_date = DT.date.today().strftime("%Y-%m-%d")):
    logger.debug("Cache miss, get_unique_locations function ran")
    logger.debug("Request date: {}".format(request_date))
    
    table_name = environment+".apt_ads_data"

    try:
        result = conn.run(
            f"""SELECT DISTINCT apt_location FROM {table_name}
            """
            )

        logger.debug(result)
    
        return [x[0] for x in result]

    except DatabaseError:
        logging.critical("A error occured: {}".format(DatabaseError))
    except Exception as e:
        logging.critical("A error occured: {}".format(e))  


@st.cache(suppress_st_warning=True)
def get_data(location = "GLOBAL",
                request_date = DT.date.today().strftime("%Y-%m-%d"),
                time_horizon = 7,
                min_price = 0,
                max_price = 999999999999,
                min_apt_size = None,
                max_apt_size = None
                ):

    logger.debug("Cache miss, get_data function ran")
    logger.debug("Request date: {}".format(request_date))
    date_lt_T = (DT.datetime.now() - DT.timedelta(days = time_horizon)).strftime("%Y-%m-%d")
    logger.debug("Start date type: {} and value: {}".format(type(date_lt_T), date_lt_T))

    logger.debug("Location list {}".format(",".join(str('{}'.format(loc)) for loc in location)))

    table_name = environment+".apt_ads_data"
    table_cols = ["ad_id", "ad_name", "ad_description", "apt_size", "apt_nb_pieces", "apt_nb_bedrooms", "apt_location",
        "apt_location_lat", "apt_location_long", "apt_price", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted", "ad_source"]
    
    query_stmt = f"""SELECT * FROM {table_name}
        WHERE ad_published_date >= :start_date
            {["AND apt_price >= :min_price" if min_price else ""][0]}
            {["AND apt_price <= :max_price" if max_price else ""][0]}
            {["AND apt_size >= :min_apt_size" if min_apt_size else ""][0]}
            {["AND apt_size <= :max_apt_size" if max_apt_size else ""][0]}
            {["AND apt_location = ANY(:location)" if location != "GLOBAL" else ""][0]}
        """

    try:
        result = conn.run(query_stmt,
            table_name = table_name,
            start_date = date_lt_T,
            min_price = min_price,
            max_price = max_price,
            min_apt_size = min_apt_size,
            max_apt_size = max_apt_size,
            location = tuple(location)
            )

        logger.debug(result)
    
        return result, table_cols

    except DatabaseError as e:
        logging.critical("A error occured: {} - {}".format(e.args, e.message))
    except Exception as e:
        logging.critical("A error occured: {}".format(e))


def clean_data(df):
    df.drop_duplicates(subset=["ad_name", "apt_location", "apt_price"], inplace= True)

    df["ad_id"] = df["ad_id"].astype("string")
    df["ad_name"] = df["ad_name"].astype("string")
    df["ad_description"] = df["ad_description"].astype("string")
    df["apt_location"] = df["apt_location"].astype("string")

    # get the postal code from the ad name
    df["apt_postal_code"] = df["apt_location"].str.extract(r'([0-9]{5})', expand= False)
    df["apt_postal_code"] = df["apt_postal_code"].astype("category")

    
    df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d %H:%M:%S", errors = "coerce").dt.strftime("%Y-%m-%d")
    df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d", errors = "coerce")

    df["ad_link"] = df["ad_link"].astype("string")
    df["ad_seller_type"] = df["ad_seller_type"].astype("category")
    df["ad_is_boosted"].fillna("false", inplace = True)
    df["ad_source"] = df["ad_source"].astype("category")

    return df

def make_link_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)

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
    df['apt_price_bins'] = pd.cut( x = df['apt_price'], bins=[0, 50000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 350000, 400000, 500000, 1000000])
    df['apt_price_per_sqm_bins'] = pd.cut( x = df['apt_price_per_sqm'], bins=[0, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 10000, 15000, 50000])
    df['apt_size_bins'] = pd.cut( x = df['apt_size'], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500])

    logger.debug(df)
    return df


def generate_cluster_chart(df, show_elbow = False):
    if len(df) <= 15:
        return None, None

    mask_no_apt_size = df["apt_size"].isna()
    mask_no_price_per_sqm = df["apt_price_per_sqm"].isna()
    mask_no_postal_code = df["apt_postal_code"].isna()
    df_null_values = df.loc[mask_no_apt_size | mask_no_price_per_sqm | mask_no_postal_code]
    df = df.loc[~df.index.isin(df_null_values.index)]

    apt_size_scaler = StandardScaler()
    df["apt_size_norm"] = apt_size_scaler.fit_transform(df[["apt_size"]])

    apt_price_per_sqm_scaler = StandardScaler()
    df["apt_price_per_sqm_norm"] = apt_price_per_sqm_scaler.fit_transform(df[["apt_price_per_sqm"]])

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit_transform(df[["apt_postal_code"]])

    kmeans = cluster.KMeans(n_clusters = 14 ,init = "k-means++")
    kmeans.fit(df[["apt_price_per_sqm_norm", "apt_size_norm"]])
    df['clusters'] = kmeans.labels_

    cluster_chart = alt.Chart(df.loc[:,("apt_size","apt_price_per_sqm","clusters")]).mark_circle().encode(
        x = "apt_size:Q",
        y = "apt_price_per_sqm:Q",
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


def generate_results_map(data):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
            lat = data["apt_location_lat"],
            lon = data["apt_location_long"],
            mode = 'markers+text',
            marker = go.scattermapbox.Marker(
                size = 17,
                color = 'rgb(255, 0, 0)',
                opacity = 1
            ),
            text = data.index,
            textposition = "bottom right",
        ))

    fig.update_layout(
        autosize = False,
        width = 1200,
        height = 800,
        hovermode = 'closest',
        showlegend = False,
        mapbox = dict(
            accesstoken = mapbox_access_token,
            bearing = 0,
            center = dict(
                lat = 45.763420,
                lon = 4.834277
            ),
            pitch = 0,
            zoom = 12,
            style = 'light'
        ),
    )

    return fig


def create_price_distr_chart(df):
    price_binned = df.loc[:,("ad_id","apt_price_bins")].groupby("apt_price_bins").count().reset_index()
    price_binned.sort_values(by = ["apt_price_bins"], inplace = True, ascending = True)
    price_binned["apt_price_bins"] = price_binned["apt_price_bins"].astype(str)

    chart = alt.Chart(price_binned).mark_bar().encode(
        x = "apt_price_bins:O",
        y = "ad_id"
    )

    return chart


def create_ppsqm_distr_chart(df):
    ppsqm_binned = df.loc[:,("ad_id","apt_price_per_sqm_bins")].groupby("apt_price_per_sqm_bins").count().reset_index()
    ppsqm_binned.sort_values(by = ["apt_price_per_sqm_bins"], inplace = True, ascending = True)
    ppsqm_binned["apt_price_per_sqm_bins"] = ppsqm_binned["apt_price_per_sqm_bins"].astype(str)

    chart = alt.Chart(ppsqm_binned).mark_bar().encode(
        x = "apt_price_per_sqm_bins:O",
        y = "ad_id"
    )

    return chart


def create_apt_size_distr_chart(df):
    size_binned = df.loc[:,("ad_id","apt_size_bins")].groupby("apt_size_bins").count().reset_index()
    size_binned.sort_values(by = ["apt_size_bins"], inplace = True, ascending = True)
    size_binned["apt_size_bins"] = size_binned["apt_size_bins"].astype(str)

    chart = alt.Chart(size_binned).mark_bar().encode(
        x = "apt_size_bins:O",
        y = "ad_id"
    )

    return chart


def generate_market_analysis(df):
    cluster_chart, elbow_chart = generate_cluster_chart(df, show_elbow= st.session_state["show_elbow_chart"])
    if cluster_chart:
        st.altair_chart(cluster_chart, use_container_width=True)
    if elbow_chart:
        st.altair_chart(elbow_chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_price_distr_chart(df))
    with col2:
        st.altair_chart(create_ppsqm_distr_chart(df))

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_apt_size_distr_chart(df))


def run_data_pipeline(data, data_cols):
    df = pd.DataFrame(data, columns = data_cols) \
        .pipe(clean_data) \
        .pipe(add_taxes_to_price) \
        .pipe(add_ppsqm) \
        .pipe(filter_ppsqm, st.session_state["ppsqm"]) \
        .pipe(deduce_apt_size) \
        .pipe(create_price_bins)

    return df


st.markdown('# Apartment search Lyon')
st.write('Please use the form in the sidebar to get results')

st.session_state['apt_data'] = None
results_placeholder = st.empty()

with st.form(key = "filter_criteria"):
    with st.sidebar:
        st.markdown('# Filters')
        start_price_slider, end_price_slider = st.select_slider('Apartment Price',
            options=[price for price in range(0,525000,25000)],
            key = "price",
            value = (0,500000))

        start_ppsqm_slider, end_ppsqm_slider = st.select_slider('Price per square meter',
            options=[price for price in range(0,25500,500)],
            key = "ppsqm",
            value = (0,25000))

        start_apt_size_slider, end_apt_size_slider = st.select_slider('Apartment size',
            options=[size for size in range(0, 305, 5)],
            key = "apt_size",
            value = (0,300))

        location_list = get_unique_locations()
        location_multiselect = st.multiselect(
                        label = "Locations",
                        options = location_list,
                        default = location_list,
                        key = "locations"
                        )

        st.markdown("***")
        elbow_chart_option = st.checkbox('Show cluster elbow chart', key= "show_elbow_chart", value = False)

        st.markdown("***")
        elbow_chart_option = st.checkbox('Show global analysis', key= "show_global_analysis", value = False)
        

        submit = st.form_submit_button(label="Submit", help=None)   

    if submit:
        for k,v in st.session_state.items():
                st.write((k,v))

        data, table_cols = get_data(time_horizon = 90,
            min_price = int(st.session_state["price"][0]),
            max_price = int(st.session_state["price"][1]),
            min_apt_size = int(st.session_state["apt_size"][0]),
            max_apt_size = int(st.session_state["apt_size"][1]),
            location = st.session_state["locations"]
            )
        
        df = run_data_pipeline(data, table_cols)

        with results_placeholder.container():
            st.write("Last data update: {}".format(get_last_data_update()))

            datatable_cols = ["ad_name", "ad_description", "apt_size", "apt_nb_pieces", "apt_location", "apt_price", "apt_price_per_sqm", "ad_link", "ad_published_date", "ad_seller_type", "ad_is_boosted"]
            st.dataframe(df.loc[:,datatable_cols].sort_values(by = "apt_price_per_sqm", ascending = True))
            st.plotly_chart(generate_results_map(df), use_container_width=True)

            st.markdown('## Market analysis')
            st.markdown('### Filtered data analysis')
            generate_market_analysis(df)

            if st.session_state["show_global_analysis"]:
                st.markdown('### Global data analysis')
                global_data, global_table_cols = get_data(location = "GLOBAL", time_horizon = 365)
                global_df = run_data_pipeline(global_data, global_table_cols)
                generate_market_analysis(global_df)