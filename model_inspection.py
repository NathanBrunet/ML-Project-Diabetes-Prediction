import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load("best_model.pkl")

# Model type
print(f"Model type: {type(model).__name__}")

# Expected features (if available)
if hasattr(model, 'feature_names_in_'):
    print("\nFeatures expected by model:")
    print(pd.DataFrame(model.feature_names_in_, columns=["Columns"]))
else:
    print("\n No feature names stored in model")

# Model parameters
print("\nModel parameters:")
print(model.get_params())

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    print("\nFeature importances:")
    importances = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importances)

# Output classes
if hasattr(model, 'classes_'):
    print("\nPredicted classes:", model.classes_)

# For sklearn pipelines
if hasattr(model, 'named_steps'):
    print("\nPipeline structure:")
    for step_name, step in model.named_steps.items():
        print(f"- {step_name} ({type(step).__name__})")