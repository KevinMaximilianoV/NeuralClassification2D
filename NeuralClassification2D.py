import numpy as np
import matplotlib.pyplot as plt

# Generar datos
np.random.seed(50)
n_samples = 200

X1 = np.random.randn(n_samples//2, 2) + np.array([2, 3])  
X2 = np.random.randn(n_samples//2, 2) + np.array([-2, 2])  

X = np.vstack((X1, X2)) 
Y = np.array([0] * (n_samples//2) + [1] * (n_samples//2)).reshape(-1, 1)  

# Sigmoide y derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_derivative(x):
    return x * (1 - x)

# Hiperparametros
input_size = 2
hidden_size = 50  # Neuronas
output_size = 1
learning_rate = 0.1
epochs = 30000

#Pesos y sesgos
W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
b1 = np.zeros((1, hidden_size))
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
b2 = np.zeros((1, output_size))

error_list = []  # Para graficar el error

# Entrenamiento
for epoch in range(epochs):
    # Propagación hacia adelante
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)

    # Calculo del error
    error = Y - predicted_output
    mse = np.mean(error**2)
    error_list.append(mse)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse}")

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    d_hidden_layer = d_predicted_output.dot(W2.T) * sigmoid_derivative(hidden_layer_output)

    # Actualizacian de pesos y sesgos
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Imprimir el error cuadratico medio final
final_mse = np.mean((Y - predicted_output) ** 2)
print(f"\nError cuadrático medio final: {final_mse}")

# Graficar
plt.plot(range(epochs), error_list)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Evolución del error durante el entrenamiento')
plt.show()

# Graficar los puntos con sus clases predichas
predictions = (predicted_output > 0.5).astype(int).flatten()

plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Clasificación de puntos en el plano')
plt.show()
