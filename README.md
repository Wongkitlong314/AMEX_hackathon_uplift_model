# AMEX_hackathon_uplift_model
This project was developed for the American Express Innovation Labs AI Hackathon (Singapore) 2024. The challenge was to maximize incremental activations for Amex Credit Card customers by recommending merchants on their online platform. Incremental activations are transactions made on merchants that customers would not have discovered without the recommendation.

Method and Approach
Uplift Modeling:

We employed uplift modeling, a causal inference technique, to estimate the incremental impact of merchant recommendations on customer activations.
Our approach focused on comparing outcomes between a treatment group (receiving recommendations) and a control group (no recommendations) to isolate the true effect of the recommendations.
Data Preparation and Preprocessing:

Class Imbalance: Addressed using Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.
Missing Values: Filled missing values with zeros to maintain data integrity.
Mixed Aggregation Levels: Normalized features to align different aggregation levels appropriately.
High Variance and Outliers: Chose not to apply standardization or dimensionality reduction techniques to avoid overfitting due to limited data representativeness.
Model Development:

X-Learner: Selected for its robustness in handling imbalanced datasets and sparse treatment group data.
Base Learners: Used Random Forest Classifier as the outcome learner and Random Forest Regressor as the effect learner.
Comparison: Evaluated against XGBoost, where Random Forest performed better for top 10 results while XGBoost excelled for the entire dataset.
Evaluation Metrics:

Top 10 Incremental Activation Rate: Focused on high potential uplift segments.
Area Under the Uplift Curve (AUUC): Measured overall model performance.
Uplift Score Distribution: Analyzed to identify promising user segments.
Limitations and Future Improvements:

Advanced Models: Explore DragonNet and Random Forest Uplift modeling.
Parameter Tuning: Conduct detailed hyperparameter optimization.
Interpretability: Utilize SHAP values and sensitivity analysis for real-world robustness.
