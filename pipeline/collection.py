from pg8000.exceptions import DatabaseError
import streamlit as st
import logging
import datetime as DT

from pipeline.connection import conn

environment = st.secrets.get('APT_SEARCH_ANALYTICS_ENV')

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
def get_ads_total(request_date = DT.date.today().strftime("%Y-%m-%d")): 
    table_name = environment+".apt_ads_data"

    try:
        result = conn.run(
            f"""SELECT COUNT(*) FROM {table_name}
            """
            )

        return result[0][0]

    except DatabaseError as e:
        logging.critical(f"A error occured: {e.args} - {e.message}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")


@st.cache(suppress_st_warning=True)
def get_price_points_total(request_date = DT.date.today().strftime("%Y-%m-%d")): 
    table_name = environment+".apt_ads_scraping_duplicate"

    try:
        result = conn.run(
            f"""SELECT COUNT(*) FROM {table_name}
            """
            )
            
        return result[0][0]

    except DatabaseError as e:
        logging.critical(f"A error occured: {e.args} - {e.message}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")

@st.cache(suppress_st_warning=True)
def get_sold_entries_total(request_date = DT.date.today().strftime("%Y-%m-%d")): 
    table_name = environment+".dvf"

    try:
        result = conn.run(
            f"""SELECT COUNT(*) FROM {table_name}
            """
            )

        return result[0][0]

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
    
        return [x[0] for x in result]

    except DatabaseError as e:
        logging.critical(f"A error occured: {e}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")  


@st.cache(suppress_st_warning=True)
def get_ad_data(table_cols = None,
                location = None,
                start_date = None,
                end_date = DT.date.today().strftime("%Y-%m-%d"),
                min_price = 0,
                max_price = 9999999,
                min_apt_size = None,
                max_apt_size = None,
                min_nb_rooms = None,
                max_nb_rooms = None
                ):

    
    logging.info(f"Start date {start_date} end date {end_date}")

    query_stmt = f"""WITH last_seen as (
        SELECT ad_id, max(ad_scraping_date) as last_date
        FROM prod.apt_ads_scraping_duplicate
        WHERE ad_scraping_date BETWEEN :start_date AND :end_date
        GROUP BY ad_id
        )

        SELECT {[",".join([f'aad.{tname}' for tname in table_cols]) if table_cols else "*"][0]}, aasd.apt_price, last_seen.last_date
        FROM prod.apt_ads_data aad
        RIGHT JOIN last_seen 
            ON aad.ad_id = last_seen.ad_id 
        LEFT JOIN prod.apt_ads_scraping_duplicate aasd
            ON aad.ad_id = aasd.ad_id
            AND aasd.ad_scraping_date = last_seen.last_date
        WHERE (aad.apt_price >= :min_price OR aasd.apt_price >= :min_price)
            AND (aad.apt_price <= :max_price OR aasd.apt_price <= :max_price)
            {["AND aad.apt_size >= :min_apt_size" if min_apt_size else ""][0]}
            {["AND aad.apt_size <= :max_apt_size" if max_apt_size else ""][0]}
            {["AND aad.apt_nb_pieces >= :min_nb_rooms" if min_nb_rooms else ""][0]}
            {["AND aad.apt_nb_pieces <= :max_nb_rooms" if max_nb_rooms else ""][0]}
            {["AND aad.apt_location_postal_code = ANY(:location)" if location else ""][0]}
            """

    try:
        result = conn.run(query_stmt,
            start_date = start_date,
            end_date = end_date,
            min_price = min_price,
            max_price = max_price,
            min_apt_size = min_apt_size,
            max_apt_size = max_apt_size,
            min_nb_rooms = min_nb_rooms,
            max_nb_rooms = max_nb_rooms,
            location = [tuple(location) if location else None][0]
            )

        table_cols.extend(["last_price", "last_seen"])

        return result, table_cols

    except DatabaseError as e:
        logging.critical(f"A error occured: {e}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")


@st.cache(suppress_st_warning=True)
def get_dvf_data(table_cols = None,
                location = None,
                start_date = None,
                end_date = DT.date.today().strftime("%Y-%m-%d"),
                min_price = None,
                max_price = None,
                min_apt_size = None,
                max_apt_size = None
                ):

    query_stmt = f"""SELECT {[",".join(table_cols) if table_cols else "*"][0]}
        FROM {environment}.dvf
        WHERE date_of_transaction BETWEEN :start_date AND :end_date
    """

    try:
        result = conn.run(query_stmt,
            start_date = start_date,
            end_date = end_date,
            min_price = min_price,
            max_price = max_price,
            min_apt_size = min_apt_size,
            max_apt_size = max_apt_size,
            location = [tuple(location) if location else None][0]
            )

        return result, table_cols

    except DatabaseError as e:
        logging.critical(f"A error occured: {e}")
    except Exception as e:
        logging.critical(f"A error occured: {e}")

