import pickle
import os

with open(os.path.join(os.getcwd(), 'model','price_model'), 'rb') as f:
    prediction_model = pickle.load(f)

def predict_prices(df):
    df["rooms_number"] = df["apt_nb_pieces"]
    model_features = ["apt_location_postal_code_OE", 'apt_size', 'rooms_number']
    df["predicted_price"] = prediction_model.predict(df[model_features]).round(0)
    df = df.drop(columns = ["rooms_number"])

    return df