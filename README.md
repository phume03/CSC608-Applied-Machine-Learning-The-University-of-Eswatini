# Applied Machine Learning #
Here we present a dive into _practical applications_ of *machine learning (ML)* in modern organizations and society. We will look into ML Theory and Methods; Data Analysis; Probability & Mathematical Statistics; Deep Learning; Probabilistic Graphical Models; Convex Optimization; Regression Analysis;  Feature engineering (e.g., basis transforms, selection , Principal Components Analysis); Classification versus Regression; Supervised methods (e.g., Naive Bayes, Decision Trees and Random Forests, Linear & Logistic Regression, Support Vector Machines (SVM), Nearest Neighbours (KNN), and Neural Networks(MLP/ANN/CNN/RNN)); Unsupervised clustering methods (e.g., k-Means, Gaussian Mixture Models, Hierarchical Clustering).



## What Is Machine Learning? ##
<details>
<summary><blockquote>"Neural networks and other forms of machine learning ultimately learn by
trial and error, one improvement at a time" ~ John Pavlus, [Quanta Magazine \(July 8, 2024\)](https://www.quantamagazine.org/what-is-machine-learning-20240708/)</blockquote></summary>
<br>
<p>By now, many people think they know what <strong>machine learning</strong> is: You “feed” computers a bunch of “training data” so that they “learn” to do things without our having to specify exactly how. But computers aren’t dogs, data isn’t kibble, and that previous sentence has way too many air quotes. What does that stuff really mean? 

Machine learning is a sub eld of <strong>articial intelligence (AI)</strong>, which explores how to computationally simulate (or surpass) humanlike intelligence. While some AI techniques (such as expert systems) use other approaches, machine learning drives most of the field’s current progress by focusing on one thing: using algorithms to automatically improve the performance of other algorithms.

Here’s how that can work in practice, for a common kind of machine learning called <strong>supervised learning</strong>. The process begins with a task — say, “recognize cats in photos.” The goal is to find a mathematical function that can accomplish the task. This function, which is called <strong>the model</strong>, will take one kind of numbers as input — in this case, digitized photographs — and transform them into more numbers as output, which might represent labels saying “cat” or “not cat.” The model has a basic mathematical form, or shape, that provides some structure for the task, but it’s not likely to produce accurate results at rst. 

Now it’s time to train the model, which is where another kind of algorithm takes over. First, a different mathematical function (called the objective) computes a number representing the current “distance” between the model’s outputs and the desired result. Then, the training algorithm uses the objective’s distance measurement to adjust the shape of the original model. It doesn’t have to “know” anything about what the model represents; it simply nudges parts of the model (called the parameters) in certain mathematical directions that minimize that distance between actual and desired output.

Once these adjustments are made, the process restarts. The updated model transforms inputs from the training examples into (slightly better) outputs, then the objective function indicates yet another (slightly better) adjustment to the model. And so on, back and forth, back and forth. After enough iterations, the trained model should be able to produce accurate outputs for most of its training examples. And here’s the real trick: It should also maintain that performance on new examples of the task, as long as they’re not too dissimilar from the training. 

Using one function to repeatedly nudge another function may sound more like busywork than “machine learning.” But that’s the whole point. Setting this mindless process in motion lets a mathematical approximation of the task emerge automatically, without human beings
having to specify which details matter. With ef cient algorithms, well chosen functions and enough examples, machine learning can create
powerful computational models that do things we have no idea how to program.

Classification and prediction tasks — like identifying cats in photos or spam in emails — usually rely on supervised machine learning, which means the training data is already sorted in advance: The photos containing cats, for example, are labeled “cat.” The training process shapes a function that can map as much of the input onto its corresponding (known) output as possible. After that, the trained model labels unfamiliar examples. 

Unsupervised learning, meanwhile, nds structure within unlabeled examples, clustering them into groups that are not speci ed in advance.
Content-recommendation systems that learn from a user’s past behavior, as well as some object-recognition tasks in computer vision, can rely on unsupervised learning. Some tasks, like the language modeling performed by systems like GPT-4, use clever combinations of supervised and unsupervised techniques known as self- and semi-supervised learning.

Finally, reinforcement learning shapes a function by using a reward signal instead of examples of desired results. By maximizing this reward through trial and error, a model can improve its performance on dynamic, sequential tasks like playing games (like chess and Go) or controlling the behavior of real and virtual agents (like self-driving cars or chatbots). 

To put these approaches into practice, researchers use a variety of exotic-sounding algorithms, from kernel machines to Q-learning. But since the 2010s, artificial neural networks have taken center stage. These algorithms — so named because their basic shape is inspired by the connections between brain cells — have succeeded at many complex tasks once considered impractical. Large language models, which use machine learning to predict the next word (or word fragment) in a string of text, are built with “deep” neural networks with billions or even trillions of parameters.

But even these behemoths, like all machine learning models, are just functions at heart — mathematical shapes. In the right context, they can be extremely powerful tools, but they also have familiar weaknesses. An “overfitted” model ts its training examples so snugly that it can’t reliably generalize, like a cat-recognizing system that fails when a photo is turned upside-down. Biases in data can be ampli ed by the training process, leading to distorted — or even unjust — results. And even when a model does work, it’s not always clear why. (Deep learning algorithms are particularly plagued by this “interpretability” problem.)

Still, the process itself is easy to recognize. Deep down, these machines all learn the same way: back and forth, back and forth.</p>
</details>

## Objectives/Outline ##
### Introduction to Applied Machine Learning ###
* Overview of machine learning landscape: Types of learning, key concepts. 
* Introduction to machine learning tools and frameworks. 
* Key concepts in data preprocessing: cleaning, feature engineering, and handling missing values. 
* Setting up Python-based environments for ML: Jupyter Notebooks, Scikit-Learn, and TensorFlow. 

### Supervised Learning \- Regression ###
* Linear regression and its assumptions. 
* Polynomial regression and regularization techniques (Ridge, Lasso). 
* Evaluating model performance: Mean Squared Error (MSE), R-squared. 
* Practical exercises: Implementing regression models on real-world datasets. 

### Supervised Learning \- Classification ###
* Introduction to classification algorithms: Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM). 
* Class imbalance and techniques to handle it (e.g., SMOTE). 
* Performance metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC. 
* Case study: Classifying customer churn or sentiment analysis. 

### Model Evaluation and Selection ###
* Cross-validation techniques: K-fold cross-validation, Leave-One-Out (LOO). 
* Hyper parameter tuning: Grid search, Random search, Bayesian optimization. 
* Model selection and comparison. 
* Overfitting and underfitting: Bias-Variance tradeoff.

### Unsupervised Learning \- Clustering ###
* Clustering algorithms: K-means, DBSCAN, Hierarchical clustering. 
* Evaluating clustering models: Silhouette score, Davies-Bouldin index. 
* Dimensionality reduction: PCA (Principal Component Analysis), t-SNE. 
* Case study: Customer segmentation. 

### Unsupervised Learning - Association Rule Learning ###
* Market Basket Analysis and Association Rule Mining. 
* Algorithms: Apriori, Eclat. 
* Applications: Recommendation systems, product bundling. 
* Practical exercise: Building a simple recommendation engine. 

### Deep Learning Fundamentals ###
* Introduction to neural networks: Perceptron, Multi-layer Perceptron (MLP). 
* Activation functions: ReLU, Sigmoid, Tanh. 
* Backpropagation and gradient descent. 
* Hands-on: Building and training a basic neural network using TensorFlow/Keras. 

### Convolutional Neural Networks (CNNs) ###
* Understanding CNNs and their applications in computer vision. 
* Layers in CNN: Convolutional layers, Pooling layers, Fully connected layers. 
* Practical application: Image classification using CNN. 
* Implementing a CNN in TensorFlow/Keras. 

### Recurrent Neural Networks (RNNs) ###
* RNNs and their applications in sequence modeling (e.g., time series, NLP). 
* Long Short-Term Memory (LSTM) networks and GRU (Gated Recurrent Unit). 
* Applications in text generation, machine translation, and stock prediction. 
* Hands-on: Building a text prediction model with LSTMs. 

### Model Deployment and Optimization ###
* Model deployment strategies: Batch processing vs. real-time. 
* Introduction to cloud platforms: AWS, Google Cloud, Azure. 
* Optimizing models for production environments (e.g., using TensorFlow Lite). 
* Case study: Deploying a machine learning model to the cloud. 

### Ethical Considerations and Fairness in ML ###
* Understanding bias and fairness in machine learning. 
* Ethics in AI: Transparency, accountability, and interpretability. 
* Fairness-aware machine learning algorithms. 
* Real-world examples and discussions on the ethical implications of ML in various industries.