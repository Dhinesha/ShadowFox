# Car Selling Price Prediction using Machine Learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("CAR SELLING PRICE PREDICTION")
print("="*60)

# Load the dataset
try:
    # Load your actual car dataset
    df = pd.read_csv(r'c:\Users\DhineshaGnanavel73\Downloads\car.csv')
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(list(df.columns))
except FileNotFoundError:
    print("Dataset not found. Creating sample dataset for demonstration.")
    # Create a sample dataset for demonstration
    np.random.seed(42)
    sample_data = {
        'fuel_type': np.random.choice(['Petrol', 'Diesel', 'CNG'], 1000),
        'years_of_service': np.random.randint(1, 20, 1000),
        'showroom_price': np.random.uniform(5, 50, 1000),
        'previous_owners': np.random.randint(0, 5, 1000),
        'kilometers_driven': np.random.uniform(10000, 200000, 1000),
        'seller_type': np.random.choice(['Dealer', 'Individual'], 1000),
        'transmission': np.random.choice(['Manual', 'Automatic'], 1000)
    }
    # Create selling price based on features (for demo)
    df = pd.DataFrame(sample_data)
    df['selling_price'] = (
        df['showroom_price'] * 0.6 - 
        df['years_of_service'] * 0.5 - 
        df['kilometers_driven'] / 10000 + 
        np.random.normal(0, 2, 1000)
    )
    df['selling_price'] = np.maximum(df['selling_price'], 1)  # Ensure positive prices
    print("Using sample dataset for demonstration.")

# Data preprocessing
print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.dropna()

# Get categorical columns automatically from the dataset
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {categorical_columns}")

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded: {col}")

# Get numerical columns (excluding target if it exists)
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
encoded_columns = [col for col in df.columns if col.endswith('_encoded')]

# Try to identify target variable (selling price)
target_col = None
possible_targets = ['selling_price', 'price', 'selling_price_lakhs', 'cost', 'value']
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

# If no standard target found, assume last numerical column is target
if target_col is None and len(numerical_columns) > 0:
    target_col = numerical_columns[-1]
    print(f"Using '{target_col}' as target variable")

# Remove target from numerical features
if target_col in numerical_columns:
    numerical_columns.remove(target_col)

# Combine all feature columns
feature_columns = numerical_columns + encoded_columns
feature_columns = [col for col in feature_columns if col in df.columns]

X = df[feature_columns]
y = df[target_col]

print(f"\nFeatures used: {feature_columns}")
print(f"Target variable: {target_col}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE METRICS")
print("="*50)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Model Accuracy: {r2*100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
print(feature_importance)

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Actual vs Predicted
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

# Plot 2: Residuals
plt.subplot(2, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Plot 3: Feature Importance
plt.subplot(2, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')

# Plot 4: Distribution of predictions
plt.subplot(2, 3, 4)
plt.hist(y_pred, bins=30, alpha=0.7, label='Predicted')
plt.hist(y_test, bins=30, alpha=0.7, label='Actual')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.legend()

# Plot 5: Error distribution
plt.subplot(2, 3, 5)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Error Distribution')

plt.tight_layout()
plt.show()

# Dynamic prediction function that adapts to your dataset
def predict_car_price(**kwargs):
    """
    Predict car selling price based on available features
    Usage: predict_car_price(feature1=value1, feature2=value2, ...)
    """
    try:
        # Create input dataframe
        input_data = pd.DataFrame([kwargs])
        
        # Handle categorical encoding
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                encoded_col = col + '_encoded'
                if encoded_col in feature_columns:
                    try:
                        input_data[encoded_col] = encoder.transform(input_data[col].astype(str))
                    except ValueError:
                        print(f"Warning: Unknown value for {col}, using default")
                        input_data[encoded_col] = 0
        
        # Add missing columns with mean values
        for col in feature_columns:
            if col not in input_data.columns:
                if col.endswith('_encoded'):
                    input_data[col] = 0  # Default for encoded categorical
                else:
                    input_data[col] = X[col].mean()  # Mean for numerical
        
        # Ensure column order
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        return prediction
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Example predictions using actual data features
print("\n" + "="*50)
print("EXAMPLE PREDICTIONS")
print("="*50)

# Show some sample predictions from actual data
if len(df) > 0:
    sample_indices = np.random.choice(df.index, min(3, len(df)), replace=False)
    
    for i, idx in enumerate(sample_indices, 1):
        sample_row = df.loc[idx]
        sample_features = {col: sample_row[col] for col in feature_columns 
                          if col in sample_row.index and not col.endswith('_encoded')}
        
        # Add categorical features (non-encoded)
        for col in categorical_columns:
            if col in sample_row.index:
                sample_features[col] = sample_row[col]
        
        if sample_features:
            prediction = predict_car_price(**sample_features)
            actual = sample_row[target_col]
            
            print(f"\n{i}. Sample Car:")
            print(f"   Actual {target_col}: {actual:.2f}")
            print(f"   Predicted {target_col}: {prediction:.2f}")
            print(f"   Error: {abs(actual - prediction):.2f}")

print("\n" + "="*50)
print("CAR PRICE PREDICTION MODEL COMPLETED!")
print("="*50)
