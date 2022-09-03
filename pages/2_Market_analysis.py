import streamlit as st
from Apartment_list import run_Kolmogorov_Smirnov_test, run_ad_data_pipeline_global_analysis, generate_market_analysis, \
    run_Kolmogorov_Smirnov_test, generate_price_distributions, generate_price_evolution

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