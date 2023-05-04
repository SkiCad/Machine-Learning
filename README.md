# Machine-Learning
#### Research Docs and Development code for machine learning model

Skin impedance is a measure of the skin's electrical resistance, and it can be used to detect anomalies or diseases such as basal cell carcinoma (BCC) and benign nevi. Machine learning models can be trained on skin impedance data to accurately classify skin abnormalities.

One popular machine learning technique used for skin impedance analysis is principal component analysis (PCA). PCA is a technique used to reduce the dimensionality of a dataset by finding the most important features or components that explain the majority of the variance in the data. In the case of skin impedance, PCA can be used to identify the most important electrical properties that distinguish between healthy and abnormal skin.

Another machine learning technique used for skin impedance analysis is the soft independent modelling of class analogy (SIMCA). SIMCA is a supervised classification technique that is based on a set of PCA models, with one PCA model for each class. SIMCA is used to determine the group membership of an unknown measurement by fitting the sample to the PCA models, and then using the distances between the unknown sample and the classes to classify the reading.

In addition to PCA and SIMCA, other machine learning models can be used for skin impedance analysis, including support vector machines (SVM), random forests, and neural networks. SVM is a type of supervised learning model that can be used for classification or regression tasks. Random forests are an ensemble learning method that can be used for both classification and regression tasks, and they are particularly useful for handling complex datasets. Neural networks are a type of deep learning model that can be used for a variety of tasks, including image recognition, speech recognition, and natural language processing.

Overall, machine learning models are powerful tools for detecting skin anomalies using impedance data. By using these models, researchers and clinicians can more accurately identify skin abnormalities, leading to earlier detection and improved patient outcomes.

**There are several machine learning models that can be used to detect skin cancer using impedance data. In this report, I will briefly explain three of them: Random Forest, Support Vector Machine, and Soft Independent Modeling of Class Analogy (SIMCA).**

1. **Random Forest:**
Random Forest is a powerful machine learning algorithm that can be used for classification and regression tasks. It is an ensemble learning method that builds multiple decision trees and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of individual trees. 

**In the context of skin cancer detection using impedance data, Random Forest can be trained on a dataset of skin impedance measurements to classify them as either benign or malignant. The algorithm can learn the complex patterns in the data that are associated with malignant skin conditions, such as asymmetrical or jagged patterns in the electrical impedance measurements. The model can then be used to predict the likelihood of malignancy in a new set of impedance measurements.**

2. **Support Vector Machine:**
Support Vector Machine (SVM) is a popular machine learning algorithm that is widely used for classification tasks. It works by finding the optimal hyperplane that separates the data points into different classes.

In the context of skin cancer detection using impedance data, SVM can be trained on a dataset of skin impedance measurements to classify them as either benign or malignant. The algorithm can learn the patterns in the data that are associated with malignant skin conditions and identify the optimal hyperplane that separates the malignant samples from the benign samples. The model can then be used to predict the likelihood of malignancy in a new set of impedance measurements.

3. **Soft Independent Modeling of Class Analogy (SIMCA):**
SIMCA is a supervised classification technique that is based on a set of PCA models. It captures the main features of a training data set for each corresponding class and defines limits around the classes. SIMCA can be used to identify the main differences between the benign and malignant skin conditions.

In the context of skin cancer detection using impedance data, SIMCA can be trained on a dataset of skin impedance measurements to capture the main features of the benign and malignant skin conditions. The algorithm can then use these models to identify the main differences between the two conditions and classify new impedance measurements as either benign or malignant.

In conclusion, Random Forest, SVM, and SIMCA are all powerful machine learning models that can be used to detect skin cancer using impedance data. These models can learn the complex patterns in the data that are associated with malignant skin conditions and identify the optimal hyperplane that separates the malignant samples from the benign samples. They offer a promising approach for early skin cancer detection, which can improve patient outcomes and reduce healthcare costs.


#### Explaining the major intituition of using machine learning model


It is difficult to say which model is better than the other in detecting skin cancer using impedance data as each model has its own strengths and weaknesses. The choice of the model depends on the specific requirements of the application and the available data. 

Logistic regression is a simple and efficient model that is often used as a baseline model for binary classification problems. It works well when the relationship between the input features and the output class is linear or can be approximated by a linear function. However, it may not work well when the relationship is non-linear or the data has complex interactions between the features.

Random forests are a popular ensemble learning method that combines multiple decision trees to improve the model's accuracy and generalization performance. It can handle non-linear relationships between features and the output class, and can handle noisy and missing data well. However, it may suffer from overfitting when the model is too complex or when there is a high dimensionality of the input data.

Support vector machines (SVMs) are a powerful model that can handle non-linear relationships between the input features and the output class using kernel functions. It works well with high-dimensional data and can handle noisy and missing data well. However, it may require careful tuning of hyperparameters, and it may not perform well when the data is imbalanced or the classes are not separable.

Soft independent modelling of class analogy (SIMCA) is a supervised classification technique that uses a set of PCA models to capture the main features of each corresponding class and define limits around the classes. It works well with high-dimensional data and can handle noisy and missing data well. However, it may not work well with non-linear relationships between features and the output class, and may not be suitable for detecting rare or novel classes.

In summary, the choice of the model depends on the specific requirements of the application and the available data. It is important to evaluate the performance of each model using appropriate metrics and to compare them against each other to select the best one for the given task.



#### Output:


![Img5](https://github.com/SkiCad/Machine-Learning/blob/main/assets/results.png)


![Img1](https://github.com/SkiCad/Machine-Learning/blob/main/assets/3d_model.gif)

![Img2](https://github.com/SkiCad/Machine-Learning/blob/main/assets/PCA.png)

![Img3](https://github.com/SkiCad/Machine-Learning/blob/main/assets/Train_loss.png)

![Img4](https://github.com/SkiCad/Machine-Learning/blob/main/assets/confusion_matrix.png)






