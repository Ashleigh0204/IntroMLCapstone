import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataCleaning import load_data, clean_target, clean_features
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from Regression import rmse

# get data
housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns = load_data()
housing_trn_clean, housing_vld_clean, housing_tst_clean = clean_features(housing_trn_df, housing_vld_df, housing_tst_df)
housing_trn_target, housing_vld_target = clean_target(housing_trn_target_ns, housing_vld_target_ns)

# define random forest
rf = RandomForestRegressor(
    n_estimators=900,
    bootstrap=True,
    max_depth=20,
    min_samples_split=10,
    random_state=42
)

# define XGBoost
xgb = XGBRegressor(
    learning_rate = 0.01,
    n_estimators=200,
    min_child_weight=2,
    subsample=1,
    colsample_bytree=0.8,
    reg_lambda=0.45,
    reg_alpha=0,
    gamma=0.5,
    seed=42
)

# define LightGBM
lgbm = LGBMRegressor(
    learning_rate=0.15,
    n_estimators=64,
    min_child_weight=2,
    num_leaves=36,
    colsample_bytree=0.8,
    reg_lambda=0.4,
    random_state=42
)

# train models
rf.fit(housing_trn_clean, housing_trn_target)
xgb.fit(housing_trn_clean, housing_trn_target)
lgbm.fit(housing_trn_clean, housing_trn_target)

# make predictions
y_hat_rf = rf.predict(housing_trn_clean)
y_hat_vld_rf = rf.predict(housing_vld_clean)
y_hat_xgb = xgb.predict(housing_trn_clean)
y_hat_vld_xgb = xgb.predict(housing_vld_clean)
y_hat_lgbm = lgbm.predict(housing_trn_clean)
y_hat_vld_lgbm = lgbm.predict(housing_vld_clean)

# make hybrid prediction
y_hat_hybrid = (1/3) * y_hat_rf + (1/3) * y_hat_xgb + (1/3) * y_hat_lgbm
y_hat_vld_hybrid = (1/3) * y_hat_vld_rf + (1/3) * y_hat_vld_xgb + (1/3) * y_hat_vld_lgbm

# calculate MSE
rmse_rf = rmse(y_hat_rf, housing_trn_target)
rmse_vld_rf = rmse(y_hat_vld_rf, housing_vld_target)
rmse_xgb = rmse(y_hat_xgb, housing_trn_target)
rmse_vld_xgb = rmse(y_hat_vld_xgb, housing_vld_target)
rmse_lgbm = rmse(y_hat_lgbm, housing_trn_target)
rmse_vld_lgbm = rmse(y_hat_vld_lgbm, housing_vld_target)

rmse_hybrid = rmse(y_hat_hybrid, housing_trn_target)
rmse_vld_hybrid = rmse(y_hat_vld_hybrid, housing_vld_target)

# print results
print(f"Random Forest RMSE:  {rmse_rf}")
print(f"Random Forest Validation RMSE:  {rmse_vld_rf}")
print(f"XGBoost RMSE:  {rmse_xgb}")
print(f"XGBoost Validation RMSE:  {rmse_vld_xgb}")
print(f"LightGBM RMSE:  {rmse_lgbm}")
print(f"LightGBM Validation RMSE:  {rmse_vld_lgbm}")
print(f"Hybrid RMSE:  {rmse_hybrid}")
print(f"Hybrid Validation RMSE:  {rmse_vld_hybrid}")

print(f"Scaled Training RMSE:  {rmse_hybrid}")
print(f"Scaled Validation RMSE:  {rmse_vld_hybrid}")

scaler = StandardScaler()
scaler.fit(housing_trn_target_ns)

original_y_hat_trn = scaler.inverse_transform(y_hat_hybrid.reshape(-1,1))
original_y_hat_vld = scaler.inverse_transform(y_hat_vld_hybrid.reshape(-1,1))
trn_rmse_ns = rmse(original_y_hat_trn, housing_trn_target_ns.reshape(-1,1))
vld_rmse_ns = rmse(original_y_hat_vld, housing_vld_target_ns.reshape(-1,1))

print(f"Training RMSE:  {trn_rmse_ns}")
print(f"Validation RMSE:  {vld_rmse_ns}")


# plot
plt.scatter(list(original_y_hat_trn.reshape(-1)) + list(original_y_hat_vld.reshape(-1)), list(housing_trn_target_ns) + list(housing_vld_target_ns))
plt.xlabel("Predicted Prices")
plt.ylabel("Actual Prices")
plt.plot([0,700000], [0,700000], color='orange', label='actual=predicted')
plt.legend()
plt.show()
        