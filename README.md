# Main Repo

# Machine Learning Techniques

# Regression:

	1.Linear Regression with scikit-learn and pandas/numpy:
	  Performed linear regression on time series data of Google’s stock from 1982 to present, achieved 98 percent accuracy and tried to forecast for next 15 days and visualize using matplotlib library.

![Time_series_prediction](https://github.com/Athul8raj/Main-Repo/blob/master/images/Time_series_prediction.png)

	2.Practical approach to Linear Regression:
	  Took random 2d data and followed an algorithmic approach to find the MSE, R^2 and ways to manipulate both these items
	3.Logistic Regression with scikit-learn and pandas/numpy:
	  Executed logistic regression method on famous Breast cancer dataset from Kaggle and achieved 99 percent accuracy to predict whether the patient has malignant tumor or otherwise

# Classification:

	1.K-Nearest Neighbors with scikit-learn:
	  Performed KNN on breast_cancer_wisconsin dataset and achieved 96 percent accuracy 
	2.Practical approach to KNN:
	  Algorithmic approach using Euclidean distance as method of classification and compared the performance to that with using scikit learn dataset
	3.Support Vector Machine using scikit-learn:
	  Linear dataset solved using SVM to achieve 98 percent accuracy
	4.Soft Margin SVM with Gaussian Kernel:
	  Random non-linear dataset analyzed using different kernels like polynomial and gaussian and their contours plotted.
	  Thanks to Mathieu Blondel, September 2010 for this code
	  
![SVM](https://github.com/Athul8raj/Main-Repo/blob/master/images/SVM%20with%20gaussian%20kernel.png)

![SVM w/ Linear Kernel](https://github.com/Athul8raj/Main-Repo/blob/master/images/SVM%20with%20Linear%20kernel.png)

	5.Practical Approach to SVM:
	  Algorithmic approach using SVM constraints and convex optimization to practically solve a random dataset
 
# Unsupervised Learning:

	1.Flat Clustering with k-Means using scikit-learn:
	  Kaggle’s Titanic dataset is analyzed and grouped into 2 clusters based on the passenger’s survival 
	2.Practical Approach to K-Means:
	  Euclidean distance along with proximity is used to cluster data points into groups
	3.Hierarchical Clustering with Mean Shift using scikit-learn:
	  Titanic dataset is clustered using bandwidth method of Mean shift
	4.Dynamic Bandwidth for Mean shift:
	  Random dataset is clustered by assigning different weights for bandwidths from data points
	  
![Mean Shift](https://github.com/Athul8raj/Main-Repo/blob/master/images/Mean%20shift%20with%20Dynamic%20bandwidth.png)

# Ensembles:

	XgBoosting with scikit-learn:
	A superconductor dataset from UCI Machine Learning Repository is fed through a XGboost regression algorithm to minimize the loss between estimated and 	predicted thermal conductivity of the superconductors. Tuning parameters included the max depth of the tree, learning parameter and number of samples in a 	leaf.

# Deep Learning Techniques:

	Artificial Neural Network with Tensorflow:
	A deep neural network comprising of 2 hidden layers of MNIST data is run against 10000 test images with 82 percent accuracy. Various hyperparameters are tuned 	to manipulate validation accuracy and loss.

	Convolutional Neural Network with Tensorflow, Keras and Tensorboard:
	Dogs_vs_Cats dataset from Kaggle is fed through a three layer Sequential Keras model neural network with convolutional, max-pooling and dense layer. The 	validation accuracy and loss is analyzed using Tensorboard. 

	Recurrent Neural Network with Tensorflow, Keras and Tensorboard:
	Dataset of various crypto-currencies are analyzed using a recurrent neural network and used for forecasting 15 days into future. A basic LSTM cell along with a 	dense network is used. Validation accuracy and loss are analyzed using Tensorboard.

# Transfer Learning Techniques:

	Using Pre-trained Model with Tensorflow, Keras and Tensorboard:
	A pre-trained CNN model using Dogs_vs_Cats dataset is used to classify horses and Human. A sequential Keras model along with 3 hidden CNN layers is fused with 	a Dense layer to correctly classify between horses and humans.
	
	Using Fine Tuning using VGG16net with Tensorflow, Keras:
	VGG16 model is used for fine tuning a classifier by adding a dense layer to the VGG16 model using Keras Sequential model and training it.The model is also augmented by ImageDataGenerator to generate images to train,test and validate the model.

# Generative Adversarial Networks(GANs):

	A GAN is fed the mnist data to a descriptor based on a CNN network and random noise is given to a Generator and is used to generate correct random image of 	mnist test data

# Flask Application with Course Info app and an anti -Plagiarism App and Memory Report Generating App

	All the python apps here make use of the ever powerful Flask framework and pipenv commands are used to create the virtaul env and the pipfile and 		pipfile.lock.

	# Memory Report App

	This App lets you handle big chunks of file in Excel or Csv format with huge memory load in terms of hundreds of GBs and output the final memory load after 		conversion. This app will soon have visualisation in addition to the report which helps user to see the data they wanted.

	This app makes advantage of Pandas library in the way how Pandas Dataframe handles data.Using this library when a file is uploaded all the heavy memory loaders 	are converted to their less memory consuming counterparts(int64 to int8, float64 to float32 and unique string to its integer "categories".

	# Defenders App

	This is a Anti-Plagiarism app which make use of an nltk library to search through the web for certain matches 	and 	give the corresponding probability of 	plagiarism.You can find the app in the URl below which is hosted on pythonanywhere.com

	http://athulrajp.pythonanywhere.com/

     	# Angular App with BS4

	This is a Angular-typescript based app with bootstrap 4 powered along with a NoSQL database (Firebase).This app displays a User interface where the user can 	register /Login through social media platforms and go through the content present over the website.

	# Software Courses App

	This flask app lets the user view the content(Software courses) after he/she logins through Social media platforms which gets registered in the database	(Postegres).If the user intends to add content into the website they can add the url and description to the wesite and through Asynchronous task scheduler 		Celery(with redis broker) to send a mail to the approver as a pdf listing the details.Once it is approved the content will be reflected on the website

