import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


class ParcorrToAdjacencyModel:
    """
    Neural network model to learn mapping from partial correlation matrices to adjacency matrices.

    The model treats the 10x10 matrices as image-like data and uses CNNs to capture spatial patterns,
    followed by fully connected layers to generate the final adjacency matrix prediction.
    """

    def __init__(
        self, input_shape=(10, 10, 1), output_shape=(10, 10), learning_rate=0.001
    ):
        """
        Initialize the model with the specified input and output shapes.

        Args:
            input_shape (tuple): Shape of input partial correlation matrices (height, width, channels)
            output_shape (tuple): Shape of output adjacency matrices (height, width)
            learning_rate (float): Learning rate for the optimizer
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and compile the neural network model.

        Returns:
            tf.keras.models.Model: Compiled Keras model
        """
        # Input layer
        input_layer = Input(shape=self.input_shape)

        # Convolutional layers to extract features from the matrix
        x = Conv2D(64, (3, 3), padding="same")(input_layer)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, (3, 3), padding="same")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        # Flatten the output
        x = Flatten()(x)

        # Dense layers to learn the mapping
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)

        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)

        # Output layer with sigmoid activation (for binary adjacency matrices)
        x = Dense(self.output_shape[0] * self.output_shape[1], activation="sigmoid")(x)
        output_layer = Reshape(self.output_shape)(x)

        # Create and compile the model
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def prepare_data(self, parcorr_matrices, adjacency_matrices):
        """
        Prepare the data for training by reshaping and normalizing.

        Args:
            parcorr_matrices (list): List of partial correlation matrices
            adjacency_matrices (list): List of adjacency matrices

        Returns:
            tuple: (X_train, X_val, y_train, y_val) - Training and validation datasets
        """
        # Convert lists to numpy arrays if they aren't already
        X = np.array(parcorr_matrices)
        y = np.array(adjacency_matrices)

        # Reshape inputs to include channel dimension if not already present
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_val, y_train, y_val

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model on the provided data.

        Args:
            X_train (np.ndarray): Training data inputs
            y_train (np.ndarray): Training data targets
            X_val (np.ndarray): Validation data inputs
            y_val (np.ndarray): Validation data targets
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training

        Returns:
            tf.keras.callbacks.History: Training history
        """
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            "best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
        )

        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
        )

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test data inputs
            y_test (np.ndarray): Test data targets

        Returns:
            tuple: (loss, accuracy) on test data
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, parcorr_matrix):
        """
        Predict adjacency matrix from a partial correlation matrix.

        Args:
            parcorr_matrix (np.ndarray): Partial correlation matrix of shape (10, 10)

        Returns:
            np.ndarray: Predicted adjacency matrix
        """
        # Reshape the input for the model
        input_data = parcorr_matrix.reshape(1, *self.input_shape)

        # Get the prediction
        prediction = self.model.predict(input_data)[0]

        # Convert probabilities to binary (0 or 1) using 0.5 as threshold
        binary_prediction = (prediction > 0.5).astype(int)

        return binary_prediction

    def plot_comparison(self, parcorr_matrix, true_adjacency, predicted_adjacency=None):
        """
        Plot the partial correlation matrix, true adjacency, and predicted adjacency.

        Args:
            parcorr_matrix (np.ndarray): Partial correlation matrix
            true_adjacency (np.ndarray): True adjacency matrix
            predicted_adjacency (np.ndarray, optional): Predicted adjacency matrix
        """
        if predicted_adjacency is None:
            predicted_adjacency = self.predict(parcorr_matrix)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot partial correlation matrix
        im0 = axes[0].imshow(parcorr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axes[0].set_title("Partial Correlation Matrix")
        plt.colorbar(im0, ax=axes[0])

        # Plot true adjacency matrix
        im1 = axes[1].imshow(true_adjacency, cmap="binary")
        axes[1].set_title("True Adjacency Matrix")

        # Plot predicted adjacency matrix
        im2 = axes[2].imshow(predicted_adjacency, cmap="binary")
        axes[2].set_title("Predicted Adjacency Matrix")

        plt.tight_layout()
        plt.show()


def generate_training_data(builder, n_samples=100):
    """
    Generate training data from the Builder class.

    Args:
        builder (Builder): Builder instance
        n_samples (int): Number of samples to generate

    Returns:
        tuple: (parcorr_matrices, adjacency_matrices)
    """
    parcorr_matrices = []
    adjacency_matrices = []

    # Get all function names
    function_names = list(builder.generated_observations.keys())

    # Iterate over each function
    for function_name in function_names:
        # Get maximum index for this function
        max_index = len(builder.generated_observations[function_name]) - 1

        # Generate samples for this function
        for i in range(min(n_samples // len(function_names), max_index + 1)):
            parcorr_matrix = builder.get_parcorr_matrix(function_name, i)
            adjacency_matrix = builder.get_adjacency_matrix(function_name, i)

            parcorr_matrices.append(parcorr_matrix.values)
            adjacency_matrices.append(adjacency_matrix.values)

    return np.array(parcorr_matrices), np.array(adjacency_matrices)
