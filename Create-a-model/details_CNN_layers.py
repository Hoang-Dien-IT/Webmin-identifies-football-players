import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()
shape = (100, 100, 1)
model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
model.add(tf.keras.layers.Activation("relu"))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(tf.keras.layers.Activation("relu"))
model.add(Dense(59))  # Giả sử len(categories) = 10
model.add(tf.keras.layers.Activation("softmax"))

model.summary()


img_path = '00000_00000.jpg'
img = image.load_img(img_path, target_size=(100, 100), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

def display_image(img, title=""):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Hiển thị hình ảnh đầu vào
display_image(img_array[0, :, :, 0], title="Input Image")

# Lấy đầu ra của từng lớp
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_array)

# Hiển thị kết quả của từng lớp
layer_names = [layer.name for layer in model.layers]
for layer_name, layer_activation in zip(layer_names, activations):
    if len(layer_activation.shape) == 4:
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // 8
        display_grid = np.zeros((size * n_cols, 8 * size))
        for col in range(n_cols):
            for row in range(8):
                channel_image = layer_activation[0, :, :, col * 8 + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image
        plt.figure(figsize=(1. / size * display_grid.shape[1],
                            1. / size * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
    else:
        print(f"Layer {layer_name} output shape: {layer_activation.shape}")
        if len(layer_activation.shape) == 2:
            print(f"Layer {layer_name} output values (first 10): {layer_activation[0][:10]}")  # Hiển thị 10 giá trị đầu tiên
            plt.figure()
            plt.title(layer_name)
            plt.plot(layer_activation[0])
            plt.show()
        elif len(layer_activation.shape) == 1:  # Lớp cuối cùng (softmax)
            print(f"Layer {layer_name} output vector: {layer_activation[0]}")
