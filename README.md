  The *Class_Balancer* python class is a comprehensive solution to the problem of imbalanced datasets. For a large variety of machine learning algorithms, class labels should typically be well-balanced to ensure highest performance when tasked with differentiation between classes. With the *Class_Balancer*, datasets are balanced by randomly oversampling from marginalized classes. However, when datasets are modified in this manner, the model learns a modified distribution of prior class probabilities. The *Class_Balancer* can be used to recover the original prior distribution by performing a transformation on prediction data.

The class functionality is as follows:
1) Instantiate a *Class_Balancer* object and denote whether the mode is 'binary' or 'multi-class' classification.
2) Call the *fit()* method, passing *Obs_IDs* as a numpy array of observation ID's and *Obs_Labels* as a one-hot encoded numpy array of your response variable.
3) Use the *sample()* method to return a balanced numpy array of observation ID's.
4) Train your model and predict class probabilities.
5) Use the *recover()* method on your prediction data to obtain estimates which reflect the true prior distribution of class probabilities.
