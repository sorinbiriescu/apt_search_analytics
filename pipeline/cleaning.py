import pandas as pd


def clean_ad_data(df):
    df.drop_duplicates(subset=["ad_name", "apt_location_postal_code", "apt_price"], inplace= True)

    df["ad_id"] = df["ad_id"].astype("string")
    df["ad_name"] = df["ad_name"].astype("string")
    
    try:
        df["ad_description"] = df["ad_description"].astype("string")
    except:
        pass

    df["apt_location_postal_code"] = df["apt_location_postal_code"].astype("string")

    try:
        df["apt_nb_pieces"].fillna(0, inplace = True)
        df["apt_nb_pieces"] = df["apt_nb_pieces"].astype(int)
    except:
        pass

    try:
        df["apt_nb_bedrooms"].fillna(0, inplace = True)
        df["apt_nb_bedrooms"] = df["apt_nb_bedrooms"].astype(int)
    except:
        pass

    df["apt_price"].fillna(0, inplace = True)
    df["apt_price"] = pd.to_numeric(df["apt_price"], downcast= 'float')

    try:
        df["apt_size"] = pd.to_numeric(df["apt_size"], downcast= 'float')
    except:
        pass
    
    try:
        df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d %H:%M:%S", errors = "coerce").dt.strftime("%Y-%m-%d")
        df["ad_published_date"] = pd.to_datetime(df["ad_published_date"], format = "%Y-%m-%d", errors = "coerce")
    except:
        pass

    try:
        df["ad_link"] = df["ad_link"].astype("string")
    except:
        pass

    try:
        df["ad_seller_type"] = df["ad_seller_type"].astype("category")
    except:
        pass

    try:
        df["ad_is_boosted"].fillna("false", inplace = True)
    except:
        pass

    try:
        df["ad_source"] = df["ad_source"].astype("category")
    except:
        pass

    try:
        df["apt_location_lat"] = df["apt_location_lat"].astype(float)
        df["apt_location_long"] = df["apt_location_long"].astype(float)
    except:
        pass

    return df


def make_link_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'