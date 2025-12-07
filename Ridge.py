import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataCleaning import load_data, clean_features, clean_target
from Regression import BaseModel, rmse, get_mini_batches
from sklearn.preprocessing import PolynomialFeatures

class L2Regression(BaseModel):
    """
        Performs polynomial regression with l2 regularization using the ordinary least squares algorithm
    
        attributes:
            w (np.ndarray): weight matrix that is inherited from OrdinaryLeastSquares
            
            degree (int): the number of polynomial degrees to include when adding
                polynomial features.
    """

    def __init__(self, degree: int, alpha: float, lamb: float, epochs: int, batch_size: int, seed:int=42, verbose=False):
        self.degree = degree
        self.alpha = alpha
        self.lamb = lamb
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    # add a biases
    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    def add_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        poly_feat = PolynomialFeatures(degree=self.degree, include_bias=False)
        return poly_feat.fit_transform(X)
        
    def fit(self, X: np.ndarray, y: np.ndarray, X_vld: np.ndarray=None, y_vld: np.ndarray=None):
        poly_X = self.add_polynomial_features(X)
        poly_X = self.add_ones(poly_X)

        if X_vld is not None:
            poly_X_vld = self.add_polynomial_features(X_vld)
            poly_X_vld = self.add_ones(poly_X_vld)
            
        # assign a random value for w
        rng = np.random.RandomState(self.seed)
        self.w = rng.random(poly_X.shape[1])
        mse_list = []
        mse_vld_list = []
        
        for e in range(self.epochs):
            sse = 0
            sse_vld = 0
            if self.verbose:
                print(f"\nEpoch:  {e}")
            batches = get_mini_batches(data_len=len(X), batch_size=self.batch_size)
            for mb in batches:
                #current weight prediction
                y_hat = poly_X[mb] @ self.w

                #update gradient
                avg_gradient = ((y_hat - y[mb]) @ poly_X[mb] + self.lamb * self.w)/len(mb)

                #update w
                self.w -= self.alpha * avg_gradient

                # calculate error
                error = (y_hat - y[mb])
                sse += np.sum(error**2)

            # calculate validation error
            if X_vld is not None and y_vld is not None:
                y_hat_vld = poly_X_vld @ self.w
                error = y_hat_vld - y_vld
                sse_vld += np.sum(error**2)
                mse_vld_list.append(sse_vld/ len(X))
                    
            mse_list.append(sse / len(X))
            if self.verbose:
                print(f"MSE:  {mse_list[-1]}")
                if X_vld is not None and y_vld is not None:
                    print(f"Validation MSE:  {mse_vld_list[-1]}")
        return mse_list, mse_vld_list

    def predict(self, X: np.ndarray) -> np.ndarray:
        poly_X = self.add_polynomial_features(X)
        poly_X = self.add_ones(poly_X)
        y_hat = poly_X @ self.w

        return y_hat

# get data
housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns = load_data()
housing_trn_clean, housing_vld_clean, housing_tst_clean = clean_features(housing_trn_df, housing_vld_df, housing_tst_df)
housing_trn_target, housing_vld_target = clean_target(housing_trn_target_ns, housing_vld_target_ns)

pr = L2Regression(
    degree=2,
    alpha=0.00005,
    lamb = 2500,
    epochs=200,
    batch_size=64,
    seed=42, 
    verbose=False
)

# fit model
mse_list, mse_vld_list = pr.fit(housing_trn_clean, housing_trn_target, housing_vld_clean, housing_vld_target)

# make predictions
y_hat_trn = pr.predict(housing_trn_clean)
y_hat_vld = pr.predict(housing_vld_clean)

# calculate error
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