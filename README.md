
Plant Disease Detection
Introduction
This project implements a deep learning model to detect and classify plant diseases from images. The model is built using the VGG19 pre-trained convolutional neural network (CNN) and is fine-tuned for this specific task. The notebook demonstrates the entire workflow, from data augmentation and preparation to model training and evaluation.

Project Structure
The project utilizes a specific directory structure for the dataset:

New Plant Diseases Dataset(Augmented)

New Plant Diseases Dataset(Augmented)

train

valid

The train directory contains 69,814 images, while the valid directory has 17,462 images, distributed across 38 different plant disease classes.

Requirements
The following Python libraries are required to run the notebook:

numpy

pandas

matplotlib

os

tensorflow

keras

Model Details
Base Model: The project uses the VGG19 model without its top layers (include_top=False).

Architecture: A custom classification head is added on top of the VGG19 base. This includes a Flatten layer followed by a Dense layer with 38 units and a softmax activation function, corresponding to the 38 classes of plant diseases.

Compilation: The model is compiled with the adam optimizer and categorical_crossentropy as the loss function, with accuracy as the evaluation metric.

Training: The model is trained for 50 epochs, with early stopping and model checkpointing to save the best performing model based on validation accuracy.

Dataset Augmentation
Data augmentation is applied to the training set using ImageDataGenerator to improve the model's robustness and prevent overfitting. The augmentation techniques used are:

preprocessing_function=preprocess_input (a function from the VGG19 module)

zoom_range=0.5

shear_range=0.3

horizontal_flip=True

How to Use
Clone the repository: Not applicable.

Download the Dataset: Ensure you have the "New Plant Diseases Dataset(Augmented)" dataset in the specified directory structure.

Install Dependencies: pip install numpy pandas matplotlib tensorflow keras

Run the Notebook: Execute the cells in the plant_disease_detection.ipynb notebook to train the model.

Results
The training history shows that the model achieved a best validation accuracy of approximately 78.71% before early stopping. The best model is saved to best_model.h5.
