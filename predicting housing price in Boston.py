import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs


%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv(r'C:\Users\wsoo\Documents\GitHub\machine-learning\projects\1. boston_housing\housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
    

print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))

#Data Explorationg
minimum_price = prices.min()
maximum_price = prices.max()
mean_price = prices.mean()
median_price = prices.median()
std_price = prices.std()

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))



#Create performance metric using R2

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    
    
    score = r2_score(y_true, y_predict)  
    
    return score

#testing on random value    
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))



#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print("Training and testing split was successful.")


# Produce learning curves for varying training set sizes and maximum depths - R2 score
vs.ModelLearning(features, prices)

#Complexitycurve - Bias Variance Tradeoff
vs.ModelComplexity(X_train, y_train)


#Model building
from sklearn.metrics import  make_scorer
from sklearn import tree
from sklearn.model_selection import GridSearchCV



def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    #Create a decision tree regressor object
    regressor = tree.DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth' : [i for i in range(1,11)]}


    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fcn = make_scorer(score_func = performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(estimator = regressor, param_grid = params, scoring = scoring_fcn, cv= cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_
    
# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))
'''
Max Depth is 4
'''


# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

               
# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print(i,price)
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

#See how prediction change on different value.     
vs.PredictTrials(features, prices, fit_model, client_data)


