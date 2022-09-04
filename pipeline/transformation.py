import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder

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
    mask_apt_size_missing = df["apt_size"].isna()
    if mask_apt_size_missing.any():
        mask_apt_price_present = df["apt_price"].notna()
        mask_apt_price_psqm_present = df["apt_price_per_sqm"].notna()
        
        df.loc[mask_apt_size_missing & mask_apt_price_present & mask_apt_price_psqm_present, ("apt_size")] = round(df["apt_price"] /df["apt_price_per_sqm"],2)
        df.loc[df["apt_size"].isnull(), ("apt_size")] = df["ad_name"].str.extract(r'([0-9]{2,3}(?=.*M))', expand = False)
        df["apt_size"] = pd.to_numeric(df["apt_size"], errors = "coerce", downcast = "float")
        df = df.loc[df["apt_size"] < 400]

        return df

    else:
        return df

def create_price_bins(df):
    df['apt_price_bins'] = pd.cut( x = df['apt_price'], bins=[0, 50000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 350000, 400000, 500000, 1000000])
    df['apt_price_per_sqm_bins'] = pd.cut( x = df['apt_price_per_sqm'], bins=[0, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 8000, 10000, 15000, 50000])
    df['apt_size_bins'] = pd.cut( x = df['apt_size'], bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500])

    return df

def align_dvf_column_names(df):
    RENAME_MAPPING = {"sell_value": "apt_price",
                     "carrez_surface": "apt_size",
                     "postal_code": "apt_location_postal_code",
                     "date_of_transaction": "last_seen"}

    return df.rename(columns = RENAME_MAPPING)


def calculate_price_delta(df):
    df["first_price"] = df["apt_price"]
    df["price_delta"] = df["apt_price"] - df["last_price"]
    df.loc[df["last_price"].notna(), ("apt_price")] = df.loc[df["last_price"].notna(), ("last_price")]
    return df

def add_record_type(df, value):
    df["record_type"] = value
    return df

def merge_ad_dvf_data(df, dvf_df):
    return pd.concat([df, dvf_df]).reset_index()

def sort_values(df):
    return df.sort_values(by = st.session_state.get("sort_option", "apt_price_per_sqm"),
        ascending = st.session_state.get("sort_direction", True))

def reset_index(df):
    return df.reset_index(drop = True)


def transform_to_cat(df, feature):
    ordinal_encoder = OrdinalEncoder()
    df[f'{feature}_OE'] = ordinal_encoder.fit_transform(df[[feature]])
    return df