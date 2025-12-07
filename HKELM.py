import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataCleaning import load_data, clean_target, clean_features
from Regression import BaseModel, rmse, get_mini_batches
from scipy.spatial.distance import cdist

class HKELM(BaseModel):
    def __init__(self, C, sigma, theta, a, b):
        self.C = C
        self.sigma = sigma
        self.theta = theta
        self.a = a
        self.b = b

    def hybrid_kernel(self, xi, xj):
        return self.theta * np.exp((-cdist(xi, xj, 'sqeuclidean')) / (2 * self.sigma**2)) + (1 - self.theta) * (xi @ xj.T + self.a)**self.b

    def fit(self, X, y):
        self.X_trn = X
        Omega = self.hybrid_kernel(X,X)
        I = np.eye(X.shape[0])
        self.w = np.linalg.pinv(Omega + I/self.C) @ y
    
    def predict(self, X):
        K = self.hybrid_kernel(X, self.X_trn)
        return K @ self.w

hkelm = HKELM(
    C = 0.2,
    sigma = 0.2,
    theta = 0.8,
    a = 0.5,
    b = 1
)

# get data
housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns = load_data()
housing_trn_clean, housing_vld_clean, housing_tst_clean = clean_features(housing_trn_df, housing_vld_df, housing_tst_df)
housing_trn_target, housing_vld_target = clean_target(housing_trn_target_ns, housing_vld_target_ns)

hkelm.fit(housing_trn_clean, housing_trn_target)

y_hat_trn = hkelm.predict(housing_trn_clean)
y_hat_vld = hkelm.predict(housing_vld_clean)

trn_rmse = rmse(y_hat_trn, housing_trn_target)
vld_rmse = rmse(y_hat_vld, housing_vld_target)

print(f"Scaled Training RMSE:  {trn_rmse}")
print(f"Scaled Validation RMSE:  {vld_rmse}")

scaler = StandardScaler()
scaler.fit(housing_trn_target_ns)

original_y_hat_trn = scaler.inverse_transform(y_hat_trn.reshape(-1,1))
original_y_hat_vld = scaler.inverse_transform(y_hat_vld.reshape(-1,1))
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