import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Melbourne Airbnb Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleansed_listings_dec18.csv")

df = load_data()
st.title("🏡 Melbourne Airbnb - Data Exploration & Explainable ML")

# Sidebar filters
st.sidebar.header("🔍 Filter Listings")
selected_room_type = st.sidebar.multiselect("Room Type", df['room_type'].unique(), default=df['room_type'].unique())
filtered_df = df[df['room_type'].isin(selected_room_type)]

# Data Overview
st.header("🔢 Dataset Preview")
st.dataframe(filtered_df.head())

# Train Deep Learning Model
@st.cache_resource
def train_deep_learning_model(df):
    tabular_features = [
        'host_is_superhost', 'latitude', 'longitude',
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'number_of_reviews', 'review_scores_rating',
        'calculated_host_listings_count'
    ]

    text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    tabular_df = df[tabular_features]
    imputer = SimpleImputer(strategy='mean')
    tabular_df_imputed = pd.DataFrame(imputer.fit_transform(tabular_df), columns=tabular_features)

    text_data = df[text_cols].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=300)
    text_features = tfidf.fit_transform(text_data)

    X_combined = hstack([tabular_df_imputed.values, text_features])
    y = (df['price'] > 100).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_combined.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        X_train.toarray(), y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    return model, imputer, tfidf, X_train, X_test

with st.spinner("Training deep learning model..."):
    model_dl, imputer_dl, tfidf_dl, X_train_dl, X_test_dl = train_deep_learning_model(df)
st.success("Deep learning model trained and ready!")

# 🔎 Search by ID and display tabular + text data
st.header("🔎 Search Listing by ID")

listing_ids = df['id'].unique()
search_id = st.text_input("Enter Listing ID")

selected_obs_combined = None
obs_tabular_imputed = None
shap_force_plot_ready = False

if search_id != "":
    try:
        search_id = int(search_id)
        if search_id in listing_ids:
            st.success(f"Listing ID {search_id} found!")

            selected_row = df[df['id'] == search_id].iloc[0]

            tabular_features = [
                'host_is_superhost', 'latitude', 'longitude',
                'accommodates', 'bathrooms', 'bedrooms', 'beds',
                'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                'calculated_host_listings_count'
            ]

            text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

            st.subheader("📋 Tabular Features")
            st.dataframe(pd.DataFrame(selected_row[tabular_features]).transpose())

            st.subheader("📝 Text Information")
            for col in text_cols:
                if pd.notnull(selected_row[col]):
                    st.markdown(f"**{col.replace('_', ' ').title()}:** {selected_row[col]}")
                else:
                    st.markdown(f"**{col.replace('_', ' ').title()}:** _No information available_")

            with st.expander("💡 Deep Learning Classification"):
                obs = df[df['id'] == search_id].copy().reset_index(drop=True)
                obs['host_is_superhost'] = obs['host_is_superhost'].map({'t': 1, 'f': 0})
                obs_tabular = obs[tabular_features]
                obs_tabular_imputed = pd.DataFrame(imputer_dl.transform(obs_tabular), columns=tabular_features)

                obs_text = obs[text_cols].fillna('').agg(' '.join, axis=1)
                obs_text_tfidf = tfidf_dl.transform(obs_text)

                selected_obs_combined = hstack([obs_tabular_imputed.values, obs_text_tfidf])

                pred_prob = model_dl.predict(selected_obs_combined.toarray())[0][0]
                pred_class = int(pred_prob > 0.5)
                label = "Expensive (Price > $100)" if pred_class else "Affordable (≤ $100)"

                st.metric(label="🔮 Prediction", value=label)
                st.caption(f"Probability of being expensive: {pred_prob:.2f}")

                shap_force_plot_ready = True

                # Show force plot here inside prediction block
                import shap
                import numpy as np
                from streamlit.components.v1 import html

                X_sample = X_train_dl.toarray()[:100]
                explainer = shap.DeepExplainer(model_dl, X_sample)

                shap_values_obs_full = explainer.shap_values(selected_obs_combined.toarray())

                if isinstance(shap_values_obs_full, list):
                    shap_values_obs_full = shap_values_obs_full[0][0]
                    expected_value = float(explainer.expected_value[0])
                else:
                    shap_values_obs_full = shap_values_obs_full[0]
                    expected_value = float(explainer.expected_value)

                tabular_part = np.array(shap_values_obs_full[:len(tabular_features)])
                text_part = np.array(shap_values_obs_full[len(tabular_features):])
                text_sum = float(text_part.sum())

                shap_values_obs_combined = np.concatenate([tabular_part.reshape(-1), np.array([text_sum])])
                combined_data = np.concatenate([obs_tabular_imputed.iloc[0].values.astype(float), np.array([0.0])])
                all_feature_names = tabular_features + ["ALL_TEXT_FEATURES"]

                explanation_force = shap.Explanation(
                    values=shap_values_obs_combined,
                    base_values=np.array([expected_value]),
                    data=combined_data.reshape(-1),
                    feature_names=all_feature_names
                )

                st.subheader("🧭 SHAP Force Plot (Selected Listing - Tabular + Summed Text)")
                st.pyplot(shap.plots.force(expected_value, shap_values_obs_combined, feature_names=all_feature_names, matplotlib=True))

        else:
            st.error("Listing ID not found in the dataset.")
    except ValueError:
        st.error("Please enter a valid numeric ID.")

# SHAP values for Deep Learning model
st.subheader("🤠 SHAP Feature Importance (Deep Learning Model)")
try:
    import shap
    import numpy as np

    X_sample = X_train_dl.toarray()[:100]
    explainer = shap.DeepExplainer(model_dl, X_sample)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_np = shap_values[0].squeeze()
    else:
        shap_values_np = shap_values.squeeze()

    tabular_features = [
        'host_is_superhost', 'latitude', 'longitude',
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'number_of_reviews', 'review_scores_rating',
        'calculated_host_listings_count'
    ]

    tabular_shap_values = shap_values_np[:, :len(tabular_features)]
    tabular_data = X_sample[:, :len(tabular_features)]

    explanation_tabular = shap.Explanation(
        values=tabular_shap_values,
        data=tabular_data,
        feature_names=tabular_features
    )

    st.subheader("📊 SHAP Bar Plot (Tabular Features Only)")
    shap.plots.bar(explanation_tabular, show=False)
    st.pyplot()

    st.subheader("📈 SHAP Beeswarm Plot (Tabular Features Only)")
    shap.plots.beeswarm(explanation_tabular, show=False)
    st.pyplot()

except Exception as e:
    raise e

st.caption("Deep Learning classifier trained to predict whether a listing price exceeds $100. SHAP explains what drives these predictions, including a single aggregated text feature.")
