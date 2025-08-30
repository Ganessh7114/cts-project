import joblib

# Load your trained pipeline
model_path = "xgboost_pipeline.pkl"  # adjust if needed
model = joblib.load(model_path)

print("✅ Model loaded successfully")

# Check what kind of object it is
print(f"\nModel type: {type(model)}")

# If it's a pipeline, list the steps
if hasattr(model, "steps"):
    print("\nPipeline steps:")
    for step_name, step_obj in model.steps:
        print(f" - {step_name}: {type(step_obj)}")

# Try to get feature names
if hasattr(model, "feature_names_in_"):
    print("\nFeature names (from model):")
    print(model.feature_names_in_)
else:
    print("\nNo direct feature_names_in_ found, checking preprocessing...")

    # If it’s a pipeline, maybe inside a ColumnTransformer
    try:
        preprocessor = dict(model.named_steps).get("preprocessor")
        if preprocessor:
            print("\nPreprocessor transformers:")
            for name, transformer, cols in preprocessor.transformers:
                print(f" - {name}: {transformer} on {cols}")
    except Exception as e:
        print("⚠️ Could not inspect preprocessor:", e)
