# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project aims to develop a predictive model for breast cancer using various machine learning classification algorithms. The algorithms evaluated include k-Nearest Neighbors (kNN), Support Vector Machine (SVM), Logistic Regression (LR), and Random Forest Classifier (RFC). The goal is to identify the most effective model based on accuracy, precision, recall, and F1-Score, thus aiding in the early detection and treatment of breast cancer.

## Key Features

- **Dataset**: Utilizes a breast cancer dataset from the Sklearn Library, consisting of several hundred human cell samples with over thirty attributes such as Radius, Texture, Concave points, and Symmetry.
- **Algorithms Implemented**: k-Nearest Neighbors (kNN), Support Vector Machine (SVM), Logistic Regression (LR), Random Forest Classifier (RFC).
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Jaccard Index.

## Highlights

- **Highest Accuracy**: Random Forest Classifier with an accuracy of 99%.
- **Highest Precision**: Random Forest Classifier with a precision of 98%.
- **Highest Recall**: Random Forest Classifier with a recall of 99%.
- **Highest F1-Score**: Random Forest Classifier with an F1-Score of 98%.

## Detailed Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | Jaccard Index |
|-----------|----------|-----------|--------|----------|---------------|
| RFC       | 0.99     | 0.98      | 0.99   | 0.98     | 0.98          |
| SVM       | 0.96     | 0.95      | 0.97   | 0.96     | 0.96          |
| LR        | 0.97     | 0.96      | 0.98   | 0.97     | 0.97          |
| KNN       | 0.95     | 0.94      | 0.96   | 0.95     | 0.94          |

![image](https://github.com/tanmaynagori14/Breast-Cancer-Analysis-And-Prediction/assets/97458530/d24713c6-e3d4-4bab-831b-1900ef3091a5)


## Confusion Matrix

The confusion matrix is a table used to describe the performance of the classification model on the test data:

| Predicted Class | Actual Class | Class=Yes (TP) | Class=No (FN) | Class=No (FP) | Class=Yes (TN) |
|-----------------|--------------|----------------|---------------|---------------|----------------|
| Class=Yes       | Class=Yes    | True Positive  | False Negative| False Positive| True Negative  |

![image](https://github.com/tanmaynagori14/Breast-Cancer-Analysis-And-Prediction/assets/97458530/58f30ce3-b708-497b-a206-b315ec5f66d7)

![image](https://github.com/tanmaynagori14/Breast-Cancer-Analysis-And-Prediction/assets/97458530/1d8208d6-f73d-403a-aa8e-4e41b5d1cd87)


## Project Team

- **Tanmay Nagori**: 2021UCD2141
- **Avi Vaswani**: 2021UCD2160
- **Deepanshu**: 2021UCD2138
- **Pravesh Gupta**: 2021UCD2119

## Institution

Netaji Subhas University of Technology (NSUT)

## Contact Information

For any queries or further information, please contact:

- **Tanmay Nagori**: tanmay14nagori@gmail.com

## Acknowledgments

We extend our heartfelt gratitude to Dr. MPS Bhatia, Professor and Head of the Department of Computer Science and Engineering at NSUT Engineering College, Dwarka, Delhi, for his exceptional guidance and support throughout this project. This project has significantly enriched our academic journey.

## References

- Sklearn Library: [Breast Cancer Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)
- International Journal of Scientific Research in Computer Science, Engineering and Information Technology: [Research Paper](http://ijsrcseit.com/)

## Code Repository

- **App.py**: [[Link to App.py](https://github.com/tanmaynagori14/Breast-Cancer-Analysis-And-Prediction/blob/main/Breast-Cancer-Predictor-master/app.py)]
- **Implementation.py**: [[Link to Implementation.py](https://github.com/tanmaynagori14/Breast-Cancer-Analysis-And-Prediction/blob/main/Breast-Cancer-Predictor-master/implementation.py)]
