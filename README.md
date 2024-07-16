# muhammadnurarasy.github.io
My Project
# CIFAR-10 Image Classification with CNN
[https://drive.google.com/file/d/1-6iwOgmYgEvWSC47TGUhTVpbbG5-ITCk/view?usp=drive_link](https://colab.research.google.com/drive/1iuoAuToI4mnoXgboDd7u-xnov412ZCX8?usp=sharing)
## Project Overview

This project demonstrates the application of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Motivation

The motivation behind this project is to gain hands-on experience with deep learning and computer vision techniques. By working on this project, we aim to:

- Understand the architecture and working of CNNs.
- Learn how to preprocess and augment image data.
- Gain insights into model training, evaluation, and hyperparameter tuning.
- Build a robust model capable of accurately classifying images.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 test images.

## Project Structure

- `cifar10_image_classification.ipynb`: Jupyter notebook containing the code for the project.
- `models/`: Directory to save trained models.
- `images/`: Directory for storing images used in the README or other documentation.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- NumPy

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/cifar10-image-classification.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter notebook:
    ```bash
    jupyter notebook cifar10_image_classification.ipynb
    ```
2. Run the cells sequentially to load the data, build the model, train it, and evaluate its performance.

## Results

The model achieves an accuracy of approximately X% on the test dataset. Below is a plot showing the training and validation accuracy over epochs:

![Accuracy Plot](images/accuracy_plot.png)

## Challenges and Solutions

**Obstacle:** Limited computational resources for training deep learning models.
**Solution:** Utilized Google Colab for free GPU access to speed up training.

## Using the Trained Model for Predictions

### Loading the Model

To use the trained model for making predictions on new images, you need to load the model first. You can load the saved model using the following code:

```python
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('cifar10_cnn_model.h5')

Making Predictions
After loading the model, you can define a function to preprocess the input image and make predictions. Here’s an example function to predict the class of a new image:
import numpy as np
import tensorflow as tf

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(image):
    # Preprocess the image
    img_array = tf.image.resize(image, (32, 32))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch axis
    
    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Get the class with the highest prediction score
    predicted_class = class_names[np.argmax(score)]
    
    return predicted_class, 100 * np.max(score)

Testing with New Images
To test the model with new images, you can use the following code:
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image

def load_and_predict_image(image_path):
    # Load the image
    img = keras_image.load_img(image_path)
    img = keras_image.img_to_array(img)
    
    # Predict the class
    predicted_class, confidence = predict_image(img)
    
    # Display the image and prediction
    plt.imshow(img.astype("uint8"))
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()

# Example usage
load_and_predict_image('path_to_your_image.jpg')

## Conclusion

This project successfully demonstrates the use of CNNs for image classification tasks. The techniques and skills learned through this project can be applied to more complex computer vision problems.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




