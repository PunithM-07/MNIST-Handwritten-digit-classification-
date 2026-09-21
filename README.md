# MNIST Handwritten Digit Classification

A deep learning project for **handwritten digit recognition using the MNIST dataset and TensorFlow/Keras**. The project trains a neural network to classify grayscale handwritten images into digits from **0 to 9**.

## 📌 Project Overview

The **MNIST (Modified National Institute of Standards and Technology)** dataset contains handwritten digit images. In this project, a neural network is trained to recognize these digits automatically.

The project includes:

* Loading and exploring the MNIST dataset
* Displaying sample handwritten digit images
* Normalizing pixel values
* Building a neural network using Keras
* Training and validating the model
* Evaluating the model using test accuracy and loss
* Visualizing training and validation performance
* Comparing actual and predicted digits
* Experimenting with a larger neural network and Dropout

## 🧠 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Jupyter Notebook

## 📂 Dataset

The project uses the **MNIST handwritten digit dataset**, which is directly loaded using TensorFlow/Keras:

```python
tf.keras.datasets.mnist
```

Each image is:

* **28 × 28 pixels**
* Grayscale
* Classified into one of **10 classes: 0–9**

The dataset is divided into training and testing data.

## ⚙️ Project Workflow

```text
MNIST Dataset
      ↓
Load Training & Testing Data
      ↓
Display Sample Images
      ↓
Normalize Pixel Values
      ↓
Build Neural Network
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Visualize Accuracy & Loss
      ↓
Predict Handwritten Digits
      ↓
Experiment with Dropout
      ↓
Compare Models
```

## 🏗️ Model Architecture

### Baseline Model

The first model uses the following architecture:

```text
Input (28 × 28)
      ↓
Flatten
      ↓
Dense Layer (128 neurons, ReLU)
      ↓
Dense Layer (10 neurons, Softmax)
      ↓
Output: Digit 0–9
```

### Model Configuration

```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(28, 28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))
```

The model is compiled using:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Metric:** Accuracy
* **Epochs:** 10
* **Validation Split:** 20%

## 🔬 Data Preprocessing

The pixel values originally range from **0 to 255**.

They are normalized to the range **0–1**:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

This helps the neural network train more effectively.

## 📊 Model Evaluation

The trained model is evaluated using the MNIST test dataset:

```python
loss, accuracy = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")
```

The project also visualizes:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

## 🔍 Digit Prediction

The trained model is tested on five handwritten images.

The project compares:

```text
Actual Digit
     vs
Predicted Digit
```

Prediction is obtained using:

```python
sample_predicted = np.argmax(
    model.predict(sample_images),
    axis=1
)
```

## 🧪 Experiment: 256 Neurons + Dropout

A second model is created to investigate the effect of increasing the number of neurons and adding Dropout.

### Modified Architecture

```text
Input (28 × 28)
      ↓
Flatten
      ↓
Dense Layer (256 neurons, ReLU)
      ↓
Dropout (0.3)
      ↓
Dense Layer (10 neurons, Softmax)
      ↓
Output: Digit 0–9
```

The modified model uses:

```python
tf.keras.layers.Dense(256, activation="relu")
tf.keras.layers.Dropout(0.3)
```

The performance of the baseline and modified models is compared using:

* Test accuracy
* Test loss
* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

## 📈 Results

The notebook automatically prints the results of both models:

```text
Baseline model  -> Test accuracy: ...
Modified model  -> Test accuracy: ...
```

The exact values depend on the training run and environment.

The comparison can be used to determine whether increasing the hidden-layer size and applying Dropout improves generalization and reduces overfitting.

## 📁 Project Structure

```text
MNIST-Handwritten-digit-classification-/
│
├── mnist_assignment_alt.ipynb
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/PunithM-07/MNIST-Handwritten-digit-classification-.git
```

### 2. Open the project folder

```bash
cd MNIST-Handwritten-digit-classification-
```

### 3. Install required libraries

```bash
pip install numpy matplotlib tensorflow jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
mnist_assignment_alt.ipynb
```

Run the cells from top to bottom.

## 💡 Key Learning Outcomes

Through this project, you can understand:

* Basics of handwritten digit classification
* MNIST dataset handling
* Image normalization
* Neural network architecture
* Dense layers
* ReLU activation
* Softmax activation
* Adam optimizer
* Sparse categorical crossentropy
* Model training and validation
* Model evaluation
* Overfitting and Dropout
* Comparing deep learning models

## 🔮 Future Improvements

The project can be extended by:

* Using Convolutional Neural Networks (CNN)
* Adding multiple hidden layers
* Using Batch Normalization
* Applying Early Stopping
* Creating a web application using Flask or Streamlit
* Allowing users to draw a digit and predict it
* Saving the trained model
* Deploying the application online

## 👨‍💻 Author

**Punith M**

Computer Science and Engineering Student

GitHub:
https://github.com/PunithM-07

## ⭐ Acknowledgement

This project uses the **MNIST handwritten digit dataset** provided through TensorFlow/Keras.

---

⭐ **If you find this project useful, consider giving the repository a star!**
