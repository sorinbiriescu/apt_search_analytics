import streamlit as st
import pandas as pd
from scipy import stats
import altair as alt

from pipeline.visualization import generate_cluster_chart, create_distr_chart
from pipeline.transformation import create_price_bins

def run_Kolmogorov_Smirnov_test():
    st.markdown("We run the KS test in order to check whether there is a signification distribution between \
        prices sold by agencies or by private individuals.")
    data = st.session_state["ad_data_global"]

    data['apt_price_per_sqm_bins'] = pd.cut( x = data['apt_price_per_sqm'], bins = range(0,12000, 250))
    stats_df_private = data.loc[data["ad_seller_type"] == "private", ("ad_id","apt_price_per_sqm_bins")] \
        .groupby("apt_price_per_sqm_bins") \
        .count() \
        .reset_index() \
        .rename(columns = {'ad_id': 'frequency'}) \
        .sort_values(by = "apt_price_per_sqm_bins", ascending = True)
    stats_df_private['pdf'] = stats_df_private['frequency'] / sum(stats_df_private['frequency'])
    stats_df_private['cdf'] = stats_df_private['pdf'].cumsum()
    stats_df_private["apt_price_per_sqm_bins"] = stats_df_private["apt_price_per_sqm_bins"].astype("string")
    
    stats_df_pro = data.loc[data["ad_seller_type"].isin(["pro", "agency"]), ("ad_id","apt_price_per_sqm_bins")] \
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
                axis=alt.Axis(labels=False,
                title = "Price / sqm")
                ),
            y='pdf'
        )

    chart_pdf_pro = alt.Chart(stats_df_pro).mark_line(interpolate='step-after', color="#FFAA00").encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_pro["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False,
                title = "Price / sqm")
                ),
            y='pdf'
        )

    chart_cdf_private = alt.Chart(stats_df_private).mark_line(interpolate='step-after').encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_private["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False,
                title = "Price / sqm")
                ),
            y='cdf'
        )

    chart_cdf_pro = alt.Chart(stats_df_pro).mark_line(interpolate='step-after', color="#FFAA00").encode(
            x= alt.X('apt_price_per_sqm_bins:O',
                sort = stats_df_pro["apt_price_per_sqm_bins"].tolist(),
                axis=alt.Axis(labels=False,
                title = "Price / sqm")
                ),
            y='cdf'
        )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_pdf_private + chart_pdf_pro, use_container_width= True)
    with col2:
        st.altair_chart(chart_cdf_private + chart_cdf_pro, use_container_width= True)
        

    return


def generate_market_analysis(scope = "local"):
    total_results = len(st.session_state[f"ad_data_{scope}"])
    start_date = st.session_state[f"ad_data_{scope}"]["last_seen"].min().strftime("%d-%m-%Y")
    end_date = st.session_state[f"ad_data_{scope}"]["last_seen"].max().strftime("%d-%m-%Y")
    st.write(f'Total data points in the analysis: **{total_results}**, from **{start_date}** to **{end_date}**')

    st.number_input("# of groups - local",
        key = f"nb_clusters_{scope}",
        min_value = 0,
        max_value = 15,
        format = "%i",
        value = 5)
    st.checkbox('Show cluster elbow chart',
        key= f"show_elbow_chart_{scope}", value = False)

    k = st.session_state[f"nb_clusters_{scope}"]
    binned_data = create_price_bins(st.session_state[f"ad_data_{scope}"])


    cluster_chart, elbow_chart = generate_cluster_chart(binned_data, k ,show_elbow= st.session_state[f"show_elbow_chart_{scope}"])


    if cluster_chart:
        st.altair_chart(cluster_chart, use_container_width=True)
    if elbow_chart:
        st.altair_chart(elbow_chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_distr_chart(binned_data,
            "apt_price_bins",
            "Total price",
            "Total price distribution")
            )
    with col2:
        st.altair_chart(create_distr_chart(binned_data,
            "apt_price_per_sqm_bins",
            "Price / sqm",
            "Price / sqm distribution"))

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(create_distr_chart(binned_data,
            "apt_size_bins",
            "Apartment size",
            "Apartment size distribution")
            )
    with col2:
        st.altair_chart(create_distr_chart(binned_data,
            "apt_location_postal_code",
            "Postal code",
            "Distribution by postal code")
            )