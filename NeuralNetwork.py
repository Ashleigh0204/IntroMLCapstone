import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from DataCleaning import load_data, clean_features, clean_target
from Regression import BaseModel, rmse, get_mini_batches

def delta_mse(y, y_hat):
    return y_hat - y

class Linear():
    @staticmethod
    def activation(z):
        return z
    
    @staticmethod
    def derivative(z):
        return np.ones(z.shape)
    
class Sigmoid():
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z):
        return Sigmoid.activation(z) * (1- Sigmoid.activation(z))

class Tanh():
    @staticmethod
    def activation(z):
        return np.tanh(z)
    
    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z)**2

class ReLU():
    @staticmethod
    def activation(z):
        return np.maximum(0,z)
    
    @staticmethod
    def derivative(z):
        z = z.copy()
        z[z>=0] = 1
        z[z<0] = 0
        return z

class Layer():
    """ Class which stores all variables required for a layer in a neural network
    
        Attributes:
            W: NumPy array of weights for all neurons in the layer
            
            b: NumPy array of biases for all neurons in the layer
            
            g: Activation function for all neurons in the layer
            
            name: Name of the layer
            
            neurons: Number of neurons in the layer
            
            inputs: Number of inputs into the layer
            
            Z: Linear combination of weights and inputs for all neurons. 
                Initialized to an empty array until it is computed and set.
                
            A: Activation output for all neurons. Initialized to an empty 
                array until it is computed and set.
    """
    def __init__(
        self, 
        W:np.array, 
        b:np.array, 
        g: object, 
        name: str=""
    ):
        self.W = W
        self.b = b
        self.g = g
        self.name = name 
        self.neurons = len(W)
        self.inputs = W.shape[1]
        self.Z = np.array([])
        self.A = np.array([])
    
    def print_info(self) -> None:
        """ Prints info for all class attributes"""
        print(f"{self.name}")
        print(f"\tNeurons: {self.neurons}")
        print(f"\tInputs: {self.inputs}")
        print(f"\tWeight shape: {self.W.shape}")
        print(f"\tBias shape: {self.b.shape}")
        print(f"\tActivation function: {self.g.__name__}")
        print(f"\tZ shape: {self.Z.shape}")
        print(f"\tA shape: {self.A.shape}")


class NeuralNetwork(BaseModel):
    """ Runs the initialization and training process for a multi-layer neural network.
        
        Attributes:
            neurons_per_layer: A list where each element represents 
                    the neurons in a layer. For example, [2, 3] would
                    create a 2 layer neural network where the hidden layer
                    has 2 neurons and the output layer has 3 neurons.
            
            g_hidden: Activation function used by ALL neurons 
                in ALL hidden layers.
                    
            g_output: Activation function used by ALL neurons
                in the output layer.
        
            alpha: learning rate or step size used by gradient descent.
                
            epochs: Number of times data is used to update the weights `self.w`.
                Each epoch means a data sample was used to update the weights at least
                once.
            
            batch_size: Mini-batch size used to determine the size of mini-batches
                if mini-batch gradient descent is used.
            
            seed: Random seed to use when initializing the layers of the neural network.

            verbose: If True, print statements inside the train() method will
                be printed.

            nn: A list of Layer class instances which define the neural network.

            avg_trn_loss_tracker: A list that tracks the average training loss per epoch. 

            avg_vld_loss_tracker: A list that tracks the average validation loss per epoch.
            
    """
    def __init__(
        self,
        neurons_per_layer: list[int],
        g_hidden: object,
        g_output: object,
        alpha: float = .001, 
        epochs: int = 1, 
        batch_size: int = 64,
        seed: int = None,
        verbose: bool = False,
    ):
        self.neurons_per_layer = neurons_per_layer
        self.g_hidden = g_hidden
        self.g_output = g_output
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self.nn = []
   

    def init_neural_network(self, n_input_features: int)-> list[Layer]:
        """ Initializes weights and biases for a multi-layer neural network 
        
            Args:
                n_input_features: Number of features the input data has
        """
        nn = []
        # Set numpy global random seed
        np.random.seed(self.seed)
        for l, neurons in enumerate(self.neurons_per_layer):
            # Set inputs to number of input features
            # for the first hidden layer
            if l == 0:
                inputs = n_input_features
            else:
                inputs = self.neurons_per_layer[l-1]
            
            # Set activation functions for the output
            # layer neurons and set the names of the nn
            if l == len(self.neurons_per_layer)-1:
                g = self.g_output
                name = f"Layer {l+1}: Output Layer"
            else:
                g = self.g_hidden
                name = f"Layer {l+1}: Hidden Layer"
            
            W = self.init_weights(neurons, inputs)
            b = np.ones([neurons,1])
            nn.append(Layer(W,b,g,name))
            
        return nn

    def init_weights(self, neurons: int, inputs: int) -> np.ndarray:
        """ Initializes weight values
        
            Args:
                neurons: Number of neurons in the layer
                
                inputs: Number of inputs to the layer
        """
        return np.random.uniform(
            low=-0.5, 
            high=0.5, 
            size=(neurons, inputs))
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        X_vld: np.ndarray = None, 
        y_vld: np.ndarray = None,
    ) -> None:
        """ Initializes and trains the defined neural network using gradient descent  
        
            Args:
                X: Training features/data 
                
                y: Training targets/labels

                X_vld: validation features/data which are used for computing the validation
                    loss after every epoch.

                y_vld: validation targets/labels which are used for computing the validation
        
        """        
        self.nn = self.init_neural_network(X.shape[1])
        mse_list = []
        mse_vld_list = []
        
        for e in range(self.epochs):
            sse = 0
            sse_vld = 0
            if self.verbose: print(f"Epoch: {e+1}")
            batches = get_mini_batches(data_len=len(X), batch_size=self.batch_size)
            for mb in batches:
                # Forward pass to get predictions
                y_hat = self.forward(X[mb])

                # Backward pass to get gradients
                self.backward(X[mb], y[mb], y_hat)

            # Calculate error
            y_hat = self.forward(X)
            mse_list.append(rmse(y_hat, y))

            # Calculate validation error
            if X_vld is not None and y_vld is not None:
                y_hat_vld = self.forward(X_vld)
                mse_vld_list.append(rmse(y_hat_vld, y_vld))
                
            if self.verbose: 
                print(f"MSE:  {mse_list[-1]}")
                if X_vld is not None and y_vld is not None:
                    print(f"Validation MSE:  {mse_vld_list[-1]}")
        return mse_list, mse_vld_list
            
        
            
    def forward(self, X:np.ndarray) -> np.ndarray:
        """ Performs the forward pass for a multi-layer neural network
    
            Args:
                X: Input features. This should be typically be the 
                    training data.
        """
        A = X.T
        for l, layer in enumerate(self.nn):
            layer.Z = layer.W @ A + layer.b
            layer.A = layer.g.activation(layer.Z)
            A = layer.A
        return A.T
    
    def backward(self, X:np.ndarray, y:np.ndarray, y_hat:np.ndarray) -> None:
        """ Performs the feedback process for a multi-layer neural network
        
            Args:
                X: Training features/data
                
                y: Training targets/labels
                
                y_hat: Training predictions (predicted targets or probabilities)
        """
        layer_index = np.arange(len(self.nn))[::-1]
        delta_A = delta_mse(y, y_hat).T
        for l, layer in zip(layer_index, self.nn[::-1]):
            if l == 0:
                A = X.T
            else:
                prev_layer = self.nn[l-1]
                A = prev_layer.A
            delta_Z = delta_A * layer.g.derivative(layer.Z)
            delta_W = delta_Z @ A.T
            W_avg_grad = delta_W / len(y)

            delta_b = delta_Z @ np.ones([len(y),1])
            b_avg_grad = delta_b / len(y)

            delta_A = layer.W.T @ delta_Z

            #updates biases and weights
            layer.b -= self.alpha * b_avg_grad
            layer.W -= self.alpha * W_avg_grad
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions using parameters learned during training.
        
            Args:
                X: Features/data to make predictions with 

            TODO:
                Finish this method by adding code to make a prediction. 
                Store the predicted labels into `y_hat`.
        """
        y_hat = self.forward(X)
        return y_hat

# get data
housing_trn_df, housing_vld_df, housing_tst_df, housing_trn_target_ns, housing_vld_target_ns = load_data()
housing_trn_clean, housing_vld_clean, housing_tst_clean = clean_features(housing_trn_df, housing_vld_df, housing_tst_df)
housing_trn_target, housing_vld_target = clean_target(housing_trn_target_ns, housing_vld_target_ns)

nn = NeuralNetwork(
    neurons_per_layer=[25,10,1],
    g_hidden=ReLU,
    g_output=Linear,
    alpha=0.002, 
    epochs=200, 
    batch_size=32,
    seed=42,
    verbose=False,
)

housing_trn_target_1 = housing_trn_target.reshape(-1,1)
housing_vld_target_1 = housing_vld_target.reshape(-1,1)

# fit model
mse_list, mse_vld_list = nn.fit(housing_trn_clean, housing_trn_target_1, housing_vld_clean, housing_vld_target_1)

# make predictions
y_hat_trn = nn.predict(housing_trn_clean)
y_hat_vld = nn.predict(housing_vld_clean)

# calculate error
trn_rmse = rmse(y_hat_trn, housing_trn_target_1)
vld_rmse = rmse(y_hat_vld, housing_vld_target_1) 

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