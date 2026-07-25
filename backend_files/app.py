
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask("Super Kart Prediction App")

# Load the trained model
model = joblib.load('tuned_random_forest_model.joblib')

# Load the scaler used for numerical features
scaler = joblib.load('scaler.joblib')

# Load the original X_train columns for consistent column alignment
original_X_train_columns = joblib.load('X_train_columns.joblib')

# Define the preprocessing steps based on the notebook
categorical_cols = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

numerical_cols_to_scale = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)

        # Replicate preprocessing steps
        # 1. Handle 'reg' in Product_Sugar_Content if present (from EDA)
        if 'Product_Sugar_Content' in query_df.columns:
            query_df['Product_Sugar_Content'] = query_df['Product_Sugar_Content'].replace('reg', 'Regular')

        # 2. One-Hot Encode Categorical Variables
        df_cat = query_df[categorical_cols]
        df_other = query_df.drop(columns=categorical_cols)
        df_cat_encoded = pd.get_dummies(df_cat, columns=categorical_cols, drop_first=True)
        processed_query_df = pd.concat([df_other, df_cat_encoded], axis=1)

        # 3. Feature Engineering: Create Store_Age
        if 'Store_Establishment_Year' in processed_query_df.columns:
            current_year = pd.Timestamp.now().year
            processed_query_df['Store_Age'] = current_year - processed_query_df['Store_Establishment_Year']
            processed_query_df = processed_query_df.drop(columns=['Store_Establishment_Year'])

        # Ensure 'Store_Age' is a numerical_col_to_scale if it exists
        if 'Store_Age' not in numerical_cols_to_scale and 'Store_Age' in processed_query_df.columns:
            numerical_cols_to_scale.append('Store_Age')

        # 4. Scale Numerical Features using the *trained* scaler
        processed_query_df[numerical_cols_to_scale] = scaler.transform(processed_query_df[numerical_cols_to_scale])

        # Align columns with training data to ensure consistency
        processed_query_df = processed_query_df.reindex(columns=original_X_train_columns, fill_value=0)

        prediction = model.predict(processed_query_df)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Ensure the model file exists for the app to load it
    if not os.path.exists('tuned_random_forest_model.joblib'):
        print("Error: 'tuned_random_forest_model.joblib' not found. Please ensure the model is saved in the same directory as app.py or update the path.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
