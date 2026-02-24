import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load Data
file_name = "data (1).csv"
df = pd.read_csv(file_name)

# 2. Feature Engineering
# Using Interaction Terms shows higher-level ML thinking to recruiters.
df['Sleep_x_Activity'] = df['Sleep Duration'] * df['Physical Activity Level']

# 3. Model Prep for OLS (Statsmodels)
# This generates the "Significance" data for LinkedIn
X = df[['Sleep Duration', 'Physical Activity Level', 'Sleep_x_Activity', 'Age']]
X = sm.add_constant(X)
y = df['Stress Level']

model = sm.OLS(y, X).fit()

# PRINT THIS AND SCREENSHOT IT:
print("\n--- OLS REGRESSION SUMMARY ---")
print(model.summary())

# 4. Residual Plot (The "Math" Media)
residuals = model.resid
predictions = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='purple', alpha=0.6, edgecolors='w')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Stress Level", fontsize=12)
plt.ylabel("Residuals (Errors)", fontsize=12)
plt.title("Model Diagnostic: Residual Plot", fontsize=14)
plt.grid(alpha=0.3)
plt.savefig("residual_plot_linkedin.png", dpi=300) # High-res for LinkedIn
print("\n--- Plot saved as residual_plot_linkedin.png ---")

# 5. ML Performance (sklearn)
X_clean = df[['Sleep Duration', 'Physical Activity Level', 'Sleep_x_Activity', 'Age']]
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

s_model = LinearRegression()
s_model.fit(X_train, y_train)
test_r2 = s_model.score(X_test, y_test)
print(f"Test Set R-squared: {test_r2:.4f}")

# 6. INTERACTIVE PREDICTION
print("\n--- LIVE STRESS PREDICTOR ---")
try:
    user_age = float(input("Enter your Age: "))
    user_sleep = float(input("Enter Sleep Duration (hrs): "))
    user_activity = float(input("Enter Physical Activity (mins): "))

    user_interaction = user_sleep * user_activity

    # Using the trained model for the interactive prediction
    user_data = [1.0, user_sleep, user_activity, user_interaction, user_age]
    user_prediction = model.predict(user_data)
    
    # Bound the result between 1 and 10
    final_score = max(1, min(10, user_prediction[0]))
    print(f"\n>>> Predicted Stress Level: {final_score:.1f} / 10")

except ValueError:
    print("Error: Input numeric values only.")
