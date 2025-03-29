import warnings
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

def inspect_model(model_path):
    try:
        model = joblib.load(model_path)
        
        print("=== MODEL INSPECTION ===")
        print(f"Model type: {type(model).__name__}")
        
        # 1. Pipeline Analysis
        if isinstance(model, Pipeline):
            print("\n[1] Pipeline Structure:")
            for step_name, step in model.named_steps.items():
                print(f"├─ {step_name}: {type(step).__name__}")
            
            # Get final classifier
            classifier = model.named_steps.get('classification', model)

        # 2. Feature Info (if available)
        print("\n[2] Feature Information:")
        if hasattr(model, 'feature_names_in_'):
            print("Expected features:")
            print(pd.DataFrame(model.feature_names_in_, columns=["Features"]))
        else:
            print("⚠️ Feature names not stored in model")

        # 3. Safe Parameters
        print("\n[3] Model Parameters (safe extract):")
        params = {}
        try:
            params = model.get_params()
        except Exception as e:
            if isinstance(model, Pipeline):
                params = {step: str(type(m)) for step, m in model.named_steps.items()}
            else:
                params = {"error": str(e)}
        print(pd.Series(params))

        # 4. Feature Importance
        print("\n[4] Feature Importance:")
        try:
            if hasattr(classifier, 'feature_importances_'):
                # Get RFECV mask and selected features
                rfecv = model.named_steps['feature_selection']
                if hasattr(rfecv, 'support_'):
                    selected_features = model.feature_names_in_[rfecv.support_]
                    importances = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': classifier.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    print(importances)
                else:
                    print("⚠️ RFECV support_ attribute not available")
            else:
                print("⚠️ Importance not available (not a tree-based model)")
        except Exception as e:
            print(f"⚠️ Could not get importance: {str(e)}")

        # 5. Classes (for classifiers)
        print("\n[5] Output Classes:")
        if hasattr(classifier, 'classes_'):
            print(classifier.classes_)
        else:
            print("⚠️ Not a classifier or classes not available")

    except Exception as e:
        print(f"❌ Critical error: {str(e)}")

# Usage
inspect_model("best_model.pkl")