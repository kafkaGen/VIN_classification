# VIN characters classification

<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

## Introduction
The project aims to develop an application capable of classifying vehicle identification numbers (VINs) based on localized handwritten symbols from scanned documents. Each vehicle has a unique VIN code that contains essential information about the machine, including manufacturing details and the manufacturer's identity. By analyzing the VIN code, the application will provide comprehensive and distinct information for each vehicle.

VINs are used to uniquely identify motor vehicles, and modern VINs consist of 17 characters. It's important to note that prior to 1980, there was no standardized format for VINs, resulting in different formats being used by various manufacturers. The current VIN format excludes the letters I, O, and Q.

The technical challenge in this project lies in developing a classifier capable of accurately distinguishing between Arabic numerals and also uppercase letters of the English alphabet (excluding I, O, and Q).

## Dataset

I have chosen three datasets: [EMNIST](https://www.kaggle.com/datasets/crawford/emnist), [HandWritten_Character](https://www.kaggle.com/datasets/vaibhao/handwritten-characters), [Devanagari Handwritten Character Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset). After examining these datasets, I determined that EMNIST was the most suitable option due to its larger number of data samples, clear class divisions, and the availability of uppercase and lowercase class distinctions in certain implementations, such as balanced-EMNIST or byclass-EMNIST. This distinction is crucial for our project as we specifically need to train the model on uppercase characters to avoid false pattern learning, such as confusing the number 1 with the letter 'l'. Therefore, we will exclude lowercase characters from our training.

To preprocess the data, the following steps were performed:

- Only the desired labels from the dataset were selected.
- A custom augmentation transformation was implemented to remove white borders after localization.
- Since the data consists of binary images (black and white or 0 and 255), the model was adapted to handle this type of input.
- All input images were resized to a standardized size of 28x28.
- Various augmentation techniques were employed, including rotation, shift, cropping, and normalization.

## Solution

To address the task, I initially experimented with a simple decision tree algorithm, specifically the LGBM Classifier, to establish a baseline and assess its adequacy as a standalone solution. However, the accuracy achieved on the balanced dataset was only 88 percent, which was deemed unsatisfactory.

Subsequently, I explored a conventional architecture comprising three convolutional neural network (CNN) layers and two fully connected (FC) layers. This solution yielded an approximate accuracy of 92 percent, but the training process was time-consuming, taking approximately 15 epochs with a batch size of 128.

I also experimented with several pre-trained models, including ResNet18, ResNet34, EfficientNet, MobileNet, and GoogLeNet. Eventually, I settled on EfficientNet_B3, which demonstrated promising results, achieving an accuracy of around 90 percent after just five epochs.

| Model Architecture | Input Size | Pretrained | Loss | Optimizer | LR Scheduler | Accuracy | Average Precision |
|--------------------|------------|------------|------|-----------|--------------|----------|----------|
| EfficientNet_b3 | 28x28 | Imagenet | CrossEntropyLoss | AdamW | ExponentialLR | 0.948 | 0.90 |

It is worth noting that the actual performance may be even better. For instance, in the EMNIST dataset, there are lowercase 'l' examples within the uppercase 'L' class, leading the model to misclassify them as the number 1 instead of 'l'. This issue affects approximately 50 percent of instances in this class. To obtain a more accurate metric, data cleaning would be required. Alternatively, it would be more appropriate to prepare data from real-life situations of VIN classification from scanned documents and train the model on this type of data directly. Additionally, some errors occur in complex cases, such as distinguishing between '2' and 'Z' or '5' and 'S', indicating that the model is approaching the level of human perception.

## Installation and Usage
To use this project and run it on your machine, you need to perform the following steps:

- Using Python Environment:
  1. Create a new environment for this project: <br/>
  `python3 -m venv /app/myenv`
  2. Activate the environment: <br/>
  `source /app/myenv/bin/activate`
  3. Install the required packages: <br/>
  `pip install -r requirements.txt`
  4. (Optional) Retrain the model: <br/>
  `python /app/train.py`
  5. Test the model on your dataset: <br/>
  `python /app/inference.py --input /mnt/test_data`

- Using Docker:
  1. Build the container: <br/>
  `docker build . -t vin`
  2. (Optional) Retrain the model: <br/>
  `docker run --rm -it vin python /app/train.py`
  3. Test the model on your dataset: <br/>
  `docker run --rm -it -v /your/path/to/test_data:/mnt/test_data vin python inference.py --input /mnt/test_data`

Please make sure to replace `/your/path/to/test_data`  with the actual paths to your test data.

> Note: The commands assume that you are in the project's root directory when executing them.

> Note: You can interrupt the training process at any time if you feel that the performance is sufficient. The best model achieved during training is already saved and can be used for further evaluation or inference.

## Author
- [Oleh Borysevych](https://github.com/kafkaGen)
- Email: borysevych.oleh87@gmail.com 

