import streamlit as st
import altair as alt
import pydeck as pdk
from pydeck.types import String
import numpy as np
from scipy import stats
import pandas as pd
import sklearn.cluster as cluster
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def generate_price_distributions():
    data = st.session_state["ad_data_global"]
    data['apt_size_bins'] = pd.cut( x = data['apt_size'], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500])
    data = data.loc[:, ("apt_size_bins", "apt_price_per_sqm", "ad_seller_type")] \
        .groupby(["apt_size_bins", "ad_seller_type"]) \
        .mean() \
        .reset_index() \
        .sort_values(by = "apt_size_bins", ascending = True)

    data["apt_size_bins"] = data["apt_size_bins"].astype("string")
    data["ad_seller_type"] = data["ad_seller_type"].astype("string")

    return alt.Chart(data).mark_bar().encode(
        x = alt.X('ad_seller_type:O', axis = alt.Axis(title = "Seller type")),
        y = alt.Y('apt_price_per_sqm:Q', axis = alt.Axis(title = "Price / sqm")),
        color = 'ad_seller_type:N',
        facet = alt.Facet('apt_size_bins:N', 
            columns= 4,
            sort = data["apt_size_bins"].tolist()
            )
        )


def calculate_price_evol_stats(data):
    m = round(data["apt_price_per_sqm"].mean(),0)
    s = data["apt_price_per_sqm"].std() 
    dof = len(data.index)-1 
    confidence = 0.95

    t_crit = np.abs(stats.t.ppf((1-confidence)/2,dof))
    se = s*t_crit/np.sqrt(len(data.index))
    ci_hi = m + se
    ci_li = m - se
    if ci_li:
        if ci_li < 0:
            ci_li = 0

    return  pd.Series([m, ci_li, ci_hi, len(data.index)], index=["mean", "ci_li", "ci_hi", "total_records"])


def generate_price_evolution(time_sample = None):
    data = st.session_state["ad_data_global"]

    data = data.loc[~data["ad_seller_type"].isin(["broker", "developer", "mandatary", "network"]),
        ("apt_location_postal_code", "last_seen", "apt_size", "apt_price_per_sqm", "record_type")]
    data = data.loc[data["apt_price_per_sqm"] >= 1000]
    # data.to_csv("export.csv")

    with st.form(key = "price_evol_form"):
        start_apt_size_slider, end_apt_size_slider = st.select_slider('Apartment size',
                options=[size for size in range(0, 305, 5)],
                key = "apt_size_price_evol_filter",
                value = [40, 60]
                )
        st.form_submit_button(label= "Apply")

    st.markdown(f"#### Between {start_apt_size_slider}$m^2$ and {end_apt_size_slider}$m^2$")
    
    data = data.loc[data["apt_size"].between(start_apt_size_slider, end_apt_size_slider)] \
        .groupby(["apt_location_postal_code", "record_type"]) \
        .resample('M', on = "last_seen") \
        .apply(calculate_price_evol_stats) \
        .reset_index()

    domain = ['ads', 'dvf']
    range_ = ['#3498db', '#27ae60']

    for index, group in data.groupby(["apt_location_postal_code"]):
        if len(group.index) <= 1:
            break

        st.markdown(f"##### Postal code: {index}")
        global_price_evolution_mean = alt.Chart(group).mark_line().encode(
            x = alt.X("last_seen:O", timeUnit='utcyearmonth', axis = alt.Axis(title = "Date")),
            y = alt.Y("mean:Q", axis = alt.Axis(title = "Mean Price / sqm")),
            color = alt.Color('record_type', scale=alt.Scale(domain=domain, range=range_))
            )

        global_price_evolution_mean_text = global_price_evolution_mean.mark_text(baseline = 'middle', dy = -10).encode(
            text = 'mean:Q',
            color = alt.value("#000000"),
        )

        global_price_evolution_ci = alt.Chart(group).mark_area().encode(
            x = alt.X("last_seen:O", timeUnit='utcyearmonth', axis = alt.Axis(title = "Date")),
            y = "ci_li:Q",
            y2 = "ci_hi:Q",
            color = alt.Color('record_type', scale=alt.Scale(domain=domain, range=range_)),
            opacity = alt.value(0.6)
            )

        global_total_records = alt.Chart(group).mark_bar(size=15).encode(
            x = alt.X("last_seen:O", timeUnit='utcyearmonth', axis = alt.Axis(title = "Date")),
            y = alt.Y("total_records:Q", axis = alt.Axis(title = "Total records")),
            color = "record_type:N"
            ).properties(
                width = 700,
                height = 100
            )

        global_total_records_text = global_total_records.mark_text(baseline='middle', dy = -10).encode(
            text = 'total_records:Q',
            color = alt.value("#000000"),
        )

        price_evol_composed_chart = global_price_evolution_ci + \
            global_price_evolution_mean + \
            global_price_evolution_mean_text                        

        price_evol_composed_chart = price_evol_composed_chart.properties(
                                        width = 700,
                                        height = 400
                                    )

        total_records_composed_chart = global_total_records + global_total_records_text

        result_chart = price_evol_composed_chart & total_records_composed_chart

        with st.container():
            st.altair_chart(result_chart, use_container_width= True)

    return


def generate_cluster_chart(df, k, show_elbow = False):
    if len(df) <= 15:
        return None, None

    mask_no_apt_size = df["apt_size"].isna()
    mask_no_price_per_sqm = df["apt_price_per_sqm"].isna()
    mask_no_postal_code = df["apt_location_postal_code"].isna()
    df_null_values = df.loc[mask_no_apt_size | mask_no_price_per_sqm | mask_no_postal_code]
    df = df.loc[~df.index.isin(df_null_values.index)]

    apt_size_scaler = StandardScaler()
    df["apt_size_norm"] = apt_size_scaler.fit_transform(df[["apt_size"]])

    apt_price_per_sqm_scaler = StandardScaler()
    df["apt_price_per_sqm_norm"] = apt_price_per_sqm_scaler.fit_transform(df[["apt_price_per_sqm"]])

    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit_transform(df[["apt_location_postal_code"]])

    kmeans = cluster.KMeans(n_clusters = k ,init = "k-means++")
    kmeans.fit(df[["apt_price_per_sqm_norm", "apt_size_norm"]])
    df['clusters'] = kmeans.labels_

    cluster_chart = alt.Chart(df.loc[:,("apt_size","apt_price_per_sqm","clusters")]).mark_circle().encode(
        x = alt.X("apt_size:Q", axis=alt.Axis(title= "Apartment size")),
        y = alt.Y("apt_price_per_sqm:Q", axis=alt.Axis(title= "Price / sqm")),
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


def create_distr_chart(data, bin_name, group_name, chart_title):
    chart_data = data.loc[:,("ad_id", bin_name)].groupby(bin_name).count().reset_index()
    chart_data.sort_values(by = [bin_name], inplace = True, ascending = True)
    chart_data[bin_name] = chart_data[bin_name].astype(str)

    return alt.Chart(chart_data).mark_bar().encode(
                x = alt.X(f"{bin_name}:O", sort = chart_data[bin_name].tolist(),axis=alt.Axis(title= group_name)),
                y = alt.Y("ad_id",  axis=alt.Axis(title= "Count"))
            ).properties(
                title = chart_title
                )