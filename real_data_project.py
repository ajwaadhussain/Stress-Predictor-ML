import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Load Data
file_name = "data (1).csv"
df = pd.read_csv(file_name)

# 2. Feature Engineering
df['Sleep_x_Activity'] = df['Sleep Duration'] * df['Physical Activity Level']

# 3. Model Prep for OLS (Statsmodels)
X = df[['Sleep Duration', 'Physical Activity Level', 'Sleep_x_Activity','Age' ]]
X = sm.add_constant(X)
y = df['Stress Level']

model = sm.OLS(y, X).fit()
print(model.summary())

# 4. Residual Plot (Saving instead of Showing to prevent blocking)
residuals = model.resid
predictions = model.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Stress Level")
plt.ylabel("Residuals (Errors)")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")  # This saves the file to your folder
print("\n--- Plot saved as residual_plot.png ---")

# 5. ML "Exam" (sklearn)
# We split the 'clean' data (without the constant) because LinearRegression adds its own
X_clean = df[['Sleep Duration', 'Physical Activity Level', 'Sleep_x_Activity','Age']]
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

s_model = LinearRegression()
s_model.fit(X_train, y_train)
print(f"Test Set Accuracy (R2): {s_model.score(X_test, y_test):.4f}")

# 6. INTERACTIVE PREDICTION
print("\n--- STRESS LEVEL PREDICTION TOOL ---")
try:
    user_age = float(input("Enter your Age: "))
    user_sleep = float(input("Enter your Sleep Duration: "))
    user_activity = float(input("Enter your Daily Activities (minutes): "))

    user_interaction = user_sleep * user_activity

    # Note: Column names MUST match the order used in sm.OLS (X)
    user_data = pd.DataFrame({
        'const': [1.0],
        'Sleep Duration' : [user_sleep],
        'Physical Activity Level': [user_activity],
        'Sleep_x_Activity': [user_interaction],
        'Age': [user_age]
    })

    user_prediction = model.predict(user_data)
    final_score = max(1, min(10, user_prediction[0]))

    print(f"\nResult: Your predicted Stress Level is {final_score:.1f} / 10")
    
    if final_score > 7:
        print("Advice: High stress predicted. Focus on recovery.")
    else:
        print("Advice: Healthy stress range. Keep it up!")

except ValueError:
    print("Error: Please enter numbers only!")