import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Preprocessing

# Load the "Boston Housing" dataset (or replace it with your desired dataset)
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Reshape the input features to match the CNN input shape
x_train = x_train.reshape(-1, x_train.shape[1], 1)
x_val = x_val.reshape(-1, x_val.shape[1], 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1)

# Step 2: Model Architecture

# Define the hybrid CNN-RNN model
def create_model(filters, kernel_size, lstm_units, dropout_rate):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = KerasRegressor(build_fn=create_model)

# Step 3: Hyperparameter Tuning

# Define the hyperparameters and their respective values to tune
param_grid = {
    'filters': [32, 64],
    'kernel_size': [3, 5],
    'lstm_units': [128, 256],
    'dropout_rate': [0.2, 0.3]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters: ", grid_result.best_params_)
best_params = grid_result.best_params_
best_model = create_model(filters=best_params['filters'], kernel_size=best_params['kernel_size'],
                          lstm_units=best_params['lstm_units'], dropout_rate=best_params['dropout_rate'])
best_model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))

# Step 4: Training

# Train the model on the training set using the best hyperparameters
best_model = grid_result.best_estimator_
best_model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))

# Step 5: Evaluation

# Evaluate the final trained model on the test set
y_pred = best_model.predict(x_test)

# Calculate evaluation metrics for Hybrid CNN-RNN model
hybrid_mse = mean_squared_error(y_test, y_pred)
hybrid_rmse = np.sqrt(hybrid_mse)
hybrid_mae = mean_absolute_error(y_test, y_pred)
hybrid_r2 = r2_score(y_test, y_pred)

# Train and evaluate the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train.squeeze(), y_train)
lr_y_pred = lr_model.predict(x_test.squeeze())

# Calculate evaluation metrics for Linear Regression model
lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)

# Train and evaluate the Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(x_train.squeeze(), y_train)
rf_y_pred = rf_model.predict(x_test.squeeze())

# Calculate evaluation metrics for Random Forest model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)

# Step 6: Compare evaluation metrics

print("Hybrid CNN-RNN Model:")
print("Mean Squared Error (MSE):", hybrid_mse)
print("Root Mean Squared Error (RMSE):", hybrid_rmse)
print("Mean Absolute Error (MAE):", hybrid_mae)
print("R-squared (R2) Score:", hybrid_r2)
print()

print("Linear Regression Model:")
print("Mean Squared Error (MSE):", lr_mse)
print("Root Mean Squared Error (RMSE):", lr_rmse)
print("Mean Absolute Error (MAE):", lr_mae)
print("R-squared (R2) Score:", lr_r2)
print()

print("Random Forest Model:")
print("Mean Squared Error (MSE):", rf_mse)
print("Root Mean Squared Error (RMSE):", rf_rmse)
print("Mean Absolute Error (MAE):", rf_mae)
print("R-squared (R2) Score:", rf_r2)

# Step 7: Visualizations

# Plot the Hybrid CNN-RNN model's predictions vs ground truth
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Ground Truth')
plt.plot(lr_y_pred, label='Hybrid CNN-RNN Predictions')
plt.xlabel('Samples')
plt.ylabel('Target Value')
plt.title('Hybrid CNN-RNN Model Predictions vs Ground Truth')
plt.legend()
plt.show()

# Plot the Linear Regression model's predictions vs ground truth
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Ground Truth')
plt.plot(y_pred, label='Linear Regression Predictions')
plt.xlabel('Samples')
plt.ylabel('Target Value')
plt.title('Linear Regression Model Predictions vs Ground Truth')
plt.legend()
plt.show()

# Plot the Random Forest model's predictions vs ground truth
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Ground Truth')
plt.plot(rf_y_pred, label='Random Forest Predictions')
plt.xlabel('Samples')
plt.ylabel('Target Value')
plt.title('Random Forest Model Predictions vs Ground Truth')
plt.legend()
plt.show()

# Step 8: Activation Map Visualization

# Extract the output of intermediate layers for visualization
layer_outputs = [layer.output for layer in best_model.model.layers[1:-1]]
activation_model = Model(inputs=best_model.model.input, outputs=layer_outputs)
activations = activation_model.predict(x_test)

# Visualize the activations of intermediate layers
for layer_activation in activations:
    n_features = layer_activation.shape[-1]
    plt.figure(figsize=(10, 6))
    plt.imshow(layer_activation[0].T.reshape(n_features, -1), cmap='viridis', aspect='auto')
    plt.xlabel('Samples')
    plt.ylabel('Features')
    plt.colorbar()
    plt.title('Activation Map')
    plt.show()

'''
# Step 9: Sensitivity Analysis

# Define the perturbation parameters to vary
perturbation_params = {
    'filters': [16, 64],
    'kernel_size': [3, 7],
    'lstm_units': [64, 256],
    'dropout_rate': [0.1, 0.4]
}

# Perform sensitivity analysis by varying the perturbation parameters
for param in perturbation_params:
    param_values = perturbation_params[param]
    mse_values = []

    for value in param_values:
        # Create and train the model with the varied parameter value
        varied_model = create_model(filters=best_params['filters'], kernel_size=best_params['kernel_size'],
                                    lstm_units=best_params['lstm_units'], dropout_rate=best_params['dropout_rate'])

        # Modify the model's parameter
        if param == 'filters':
            conv1d_layer = varied_model.layers[0]
            conv1d_layer.filters = value
            # Update the filters in the Conv1D layer
            conv1d_layer.build((None, x_train.shape[1], 1))
            # Update the subsequent LSTM layer
            lstm_layer = LSTM(units=value, return_sequences=True)
            lstm_layer.build((None, None, conv1d_layer.filters))
            varied_model.layers[2] = lstm_layer
            varied_model.layers[3] = LSTM(units=value)

        # Train the model
        varied_model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val), verbose=0)

        # Evaluate the model
        y_pred_varied = varied_model.predict(x_test)
        mse_varied = mean_squared_error(y_test, y_pred_varied)
        mse_values.append(mse_varied)

    # Plot the sensitivity analysis results
    plt.figure(figsize=(8, 6))
    plt.plot(param_values, mse_values, 'bo-')
    plt.xlabel(param)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Sensitivity Analysis: {param}')
    plt.grid(True)
    plt.show()
'''