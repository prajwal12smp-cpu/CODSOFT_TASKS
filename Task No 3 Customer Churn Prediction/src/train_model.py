from churn_pipeline import train_model_pipeline


if __name__ == "__main__":
    comparison_df, tuned_models, best_pipeline = train_model_pipeline()
    print("\n=== Model Comparison ===\n")
    print(comparison_df.to_string(index=False))
    print("\nTraining completed successfully.")
