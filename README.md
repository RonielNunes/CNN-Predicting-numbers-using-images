# CNN-Predicting-numbers-using-images

Este código cria uma arquitetura simples de CNN para um problema de classificação usando o conjunto de dados MNIST, que contém dígitos escritos à mão. Este modelo possui duas camadas convolucionais, seguidas por camadas de pooling, camadas totalmente conectadas e uma camada de saída. Para treinar e testar o modelo, você precisará de dados do conjunto MNIST e realizar operações de treinamento e avaliação.


Este código faz o seguinte:

- Carrega os dados do MNIST usando mnist.load_data()

- Realiza o pré-processamento dos dados, normalizando os valores de pixel para o intervalo (0,1)

- Define a arquitetura da NN.

- Compila o modelo especificando o otimizador, a função de perda e a métrica de interesse.

- Treina o modelo usando model.fit() com 5 épocas e um tamanho de lote de 64, usando os dados de valid6. ação.

- Avalia o desempenho do modelo nos dados de teste usando model.evaluate() e imprime a acurácia.

```Python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Verifique se a GPU está sendo usada
physical_devices = tf.config.list_physical_devices('GPU')
print("GPUs disponíveis:", physical_devices)

# Carregamento dos dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pré-processamento dos dados
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Definição do modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Avaliação do desempenho do modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de testes: {test_accuracy}")

# Exiba a arquitetura do modelo
model.summary()

```

```Python
import matplotlib.pyplot as plt

# Escolha um índice de exemplo do conjunto de testes
index = 2  # Altere o índice conforme desejado

# Faça a previsão para uma única imagem
single_image = x_test[index]
single_image = single_image.reshape(1, 28, 28, 1)  # Adapte o formato para a previsão
prediction = model.predict(single_image)

# Exiba a imagem
plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.axis('off')
plt.show()

# Resultado da previsão
predicted_class = prediction.argmax()
print(f"Classe prevista: {predicted_class}")

```

```Python
for i in range(0,10):
    # Escolha um índice de exemplo do conjunto de testes
    index = i  # Altere o índice conforme desejado
    
    # Faça a previsão para uma única imagem
    single_image = x_test[index]
    single_image = single_image.reshape(1, 28, 28, 1)  
    prediction = model.predict(single_image)
    
    # Exiba a imagem
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Resultado da previsão
    predicted_class = prediction.argmax()
    print(f"Classe prevista: {predicted_class}")
```

## **Salvar o modelo**

```Python
from tensorflow.keras.models import save_model
from datetime import datetime

# Obtém a data e hora atuais
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Nome do arquivo com data e hora
model_name = f"modelo_{current_time}.h5"
print(f'Nome do modelo: {model_name}')


# Salva o modelo com o nome contendo a data e hora
model.save(model_name)
```

## **Para carregar o modelo:**

```Python
from tensorflow.keras.models import load_model

# Carrega o modelo a partir do arquivo
loaded_model = load_model('modelo_20231218_211054.h5')
```