import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataCleaning import load_data, clean_features, clean_target
from Regression import BaseModel, rmse, get_mini_batches

class LinearRegression(BaseModel):
    '''
    Performs regression using least mean squares (gradient descent)

    attributes:
        w (np.ndarray): weight matrix

        alpha (float): learning rate or step size

        epochs (int): Number of epochs to run for mini-batch gradient descent

        batch_size (int):  batch size for mini-batch gradient descent

        seed (int): Seed to be used for NumPy's RandomState class or universal seed np.random.seed() function.
    '''
    
    def __init__(self, alpha, epochs, batch_size, seed=42, verbose=False):
        self.w = None
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    # add a biases
    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X: np.ndarray, y: np.ndarray, X_vld: np.ndarray=None, y_vld: np.ndarray=None):
        # add bias
        X = self.add_ones(X)
        if X_vld is not None:
            X_vld = self.add_ones(X_vld)
        # assign a random value for w
        rng = np.random.RandomState(self.seed)
        self.w = rng.random(X.shape[1])
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
                y_hat = X[mb] @ self.w

                #update gradient
                avg_gradient = ((y_hat-y[mb]) @ X[mb])/len(mb)

                #update w
                self.w -= self.alpha * avg_gradient

                # calculate error
                error = (y_hat - y[mb])
                sse += np.sum(error**2)

            # calculate validation error
            if X_vld is not None and y_vld is not None:
                y_hat_vld = X_vld @ self.w
                error = y_hat_vld - y_vld
                sse_vld += np.sum(error**2)
                mse_vld_list.append(sse_vld/ len(X_vld))
                    
            mse_list.append(sse / len(X))
            if self.verbose:
                print(f"MSE:  {mse_list[-1]}")
                if X_vld is not None and y_vld is not None:
                    print(f"Validation MSE:  {mse_vld_list[-1]}")
        return mse_list, mse_vld_list

    def predict(self, X: np.ndarray):
        X = self.add_ones(X)
        y_hat = X @ self.w

        return y_hat

# get data
housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns = load_data()
housing_trn_clean, housing_vld_clean, housing_tst_clean = clean_features(housing_trn_df, housing_vld_df, housing_tst_df)
housing_trn_target, housing_vld_target = clean_target(housing_trn_target_ns, housing_vld_target_ns)

lr = LinearRegression(
    alpha=0.01,
    epochs=100,
    batch_size=32,
    seed=42, 
    verbose=False
)

# fit model
mse_list, mse_vld_list = lr.fit(housing_trn_clean, housing_trn_target, housing_vld_clean, housing_vld_target)

# make predictions
y_hat_trn = lr.predict(housing_trn_clean)
y_hat_vld = lr.predict(housing_vld_clean)

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
