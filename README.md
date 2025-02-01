# Pneumonia X-ray Detection Flask

This is a simple web application that uses a trained model to predict if a person has pneumonia based on their chest x-ray image. The model was trained using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## Installation

1. Clone the repository
    ```bash
    git clone https://github.com/sprakhar778/Meta-Diagnosis.git
    ```

2. Create a virtual environment
    ```bash
    python -m venv env
    ```

3. Activate the virtual environment
    ```bash
    # Windows
    env\Scripts\activate

    # macOS & Linux
    source env/bin/activate
    ```

4. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

5. Run the application
    ```bash
    python app.py
    ```

6. Open your browser and go to http://127.0.0.1:5000

## Usage

1. Upload a chest x-ray image (For a sample image, you can use the **NORMAL** or **PNEUMONIA** image file in the `/data/test/` directory)
2. Click the `Predict` button
3. View the prediction result

## Model

You can find the model in the `/model` directory. The model was trained using a Convolutional Neural Network (CNN) with the following architecture:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 150, 150, 32)      320       
_________________________________________________________________
batch_normalization (BatchNo (None, 150, 150, 32)      128       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 75, 75, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 75, 64)        18496     
_________________________________________________________________
dropout (Dropout)            (None, 75, 75, 64)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 75, 75, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 38, 38, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 38, 38, 64)        36928     
_________________________________________________________________
batch_normalization_2 (Batch (None, 38, 38, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 19, 19, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 19, 19, 128)       73856     
...
Total params: 1,246,401
Trainable params: 1,245,313
Non-trainable params: 1,088
_________________________________________________________________
```

The model was trained using the following hyperparameters:

- Optimizer: Adam
- Loss function: Binary Crossentropy
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 12
- Early stopping: 5
- Image size: 150x150
- Training set size: 5216 images
- Validation set size: 16 images
- Test set size: 624 images
- Training time: 1 hour
- Accuracy: 0.90
- AUC: 0.87.5
- F1 score: 0.86
- Precision: 0.89
- Recall: 0.87

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

