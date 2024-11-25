import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='relu'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)  # hidden layer bias

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(output_dim)  # output layer bias

        self.gradients = {}

    def activation(self, x):
        # Define activation function
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")
    
    def activation_derivative(self, x):
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)  # Correct sigmoid derivative with pre-activation x
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    
    def forward(self, X):
        # Forward pass through the network
        self.Z1 = np.dot(X, self.W1) + self.b1  # input to hidden
        self.A1 = self.activation(self.Z1)  # activation on hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # hidden to output
        self.A2 = self.sigmoid(self.Z2)  # output layer (sigmoid for binary classification)
        return self.A2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cross_entropy_loss(self, y_pred, y_true):
        # Cross-entropy loss for binary classification
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))
        return loss
    
    def backward(self, X, y):
        # Backward pass to compute gradients
        m = X.shape[0]
        
        # Compute gradients for output layer
        output_error = self.A2 - y  # derivative of loss w.r.t. output
        dW2 = np.dot(self.A1.T, output_error) / m
        db2 = np.sum(output_error, axis=0) / m
        
        # Compute gradients for hidden layer
        hidden_error = np.dot(output_error, self.W2.T) * self.activation_derivative(self.Z1)  # derivative of tanh/sigmoid
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0) / m
        
        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['b1'] = db1
        self.gradients['W2'] = dW2
        self.gradients['b2'] = db2
        
        # Update weights and biases using gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward functions
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Graph 1: The learned features and the decision hyperplane in the hidden space
    hidden_features = mlp.A1  # Get activations of the hidden layer (features)

    # Plot the data points in the hidden layer space
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Decision Hyperplane in Hidden Layer")

    # Create a grid for visualizing the decision hyperplane in the hidden layer
    x_vals = np.linspace(min(hidden_features[:, 0]), max(hidden_features[:, 0]), 20)
    y_vals = np.linspace(min(hidden_features[:, 1]), max(hidden_features[:, 1]), 20)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    # Extract the 3D coordinates for the red/blue points
    x_hidden = hidden_features[:, 0]  # h1
    y_hidden = hidden_features[:, 1]  # h2
    z_hidden = hidden_features[:, 2]  # h3

    # Create a grid in hidden space for interpolation
    x_grid = np.linspace(x_hidden.min(), x_hidden.max(), 30)
    y_grid = np.linspace(y_hidden.min(), y_hidden.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate to create the surface (z_hidden as values)
    Z_surface = griddata(
        (x_hidden, y_hidden),  # Points in 2D hidden space (h1, h2)
        z_hidden,              # Values to interpolate (h3)
        (X_grid, Y_grid),      # Grid in hidden space
        method='linear'        # Interpolation method
    )

    # Plot the interpolated surface as the manifold
    ax_hidden.plot_surface(
        X_grid, Y_grid, Z_surface,
        color='blue', alpha=0.2, edgecolor='none'
    )

    # Decision Hyperplane in the Hidden Space
    if np.abs(mlp.W2[2, 0]) > 1e-9:  # Avoid division by zero
        Z_hyperplane = -(mlp.W2[0, 0] * X_grid + mlp.W2[1, 0] * Y_grid + mlp.b2[0]) / mlp.W2[2, 0]
        ax_hidden.plot_surface(
            X_grid, Y_grid, Z_hyperplane, color='brown', alpha=0.5, edgecolor='none'
        )

    # Set axis labels for clarity
    ax_hidden.set_xlabel('Hidden Feature 1')
    ax_hidden.set_ylabel('Hidden Feature 2')
    ax_hidden.set_zlabel('Hidden Feature 3')

    # Graph 2: The decision boundary in the input space (only data points and decision regions)
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.forward(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision regions (contour) without any lines, only colors
    ax_input.contourf(xx, yy, Z, cmap=ListedColormap(['#AAAAFF', '#FFAAAA']), alpha=0.5)
    
    # Plot the data points on the decision boundary graph
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k', s=50)
    ax_input.set_title("Decision Boundary in Input Space")

    # Remove the axis lines for a cleaner look
    ax_input.set_xlabel("Feature 1")
    ax_input.set_ylabel("Feature 2")

    # Graph 3: Visualizing gradients with varying edge thickness
    ax_gradient.set_title("Gradient Magnitude")
    
    # Define the coordinates for the nodes in the graph
    node_coords = {
        'x1': (0.0, 0.0),
        'x2': (0.0, 1.0),
        'h1': (0.5, 0.0),
        'h2': (0.5, 0.5),
        'h3': (0.5, 1.0),
        'y': (1.0, 0.5)
    }
    
    # Plot nodes as circles
    for node, (x, y) in node_coords.items():
        ax_gradient.add_patch(Circle((x, y), radius=0.05, color='b', alpha=0.5))

    # Define the edges and visualize with gradients
    edges = [
        ('x1', 'h1', mlp.gradients['W1'][0, 0]), ('x1', 'h2', mlp.gradients['W1'][0, 1]), ('x1', 'h3', mlp.gradients['W1'][0, 2]),
        ('x2', 'h1', mlp.gradients['W1'][1, 0]), ('x2', 'h2', mlp.gradients['W1'][1, 1]), ('x2', 'h3', mlp.gradients['W1'][1, 2]),
        ('h1', 'y', mlp.gradients['W2'][0, 0]), ('h2', 'y', mlp.gradients['W2'][1, 0]), ('h3', 'y', mlp.gradients['W2'][2, 0])
    ]
    
    # Compute and visualize the gradients for edges
    for node1, node2, gradient in edges:
        grad_magnitude = np.abs(gradient) * 100  # Scale for visibility
        ax_gradient.plot([node_coords[node1][0], node_coords[node2][0]], 
                         [node_coords[node1][1], node_coords[node2][1]],
                         linewidth=grad_magnitude, color='gray')

    # Set x and y axes for clarity
    ax_gradient.set_xlabel("Input-Hidden-Output Progression")
    ax_gradient.set_ylabel("Nodes Alignment")

    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.axis('on')  # Enable axis

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
