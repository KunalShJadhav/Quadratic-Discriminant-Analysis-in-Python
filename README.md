# QUADRATIC DISCRIMINANT ANALYSIS BY PRINCIPAL OF MISSCLASSIFICATION
This Method Uses Principal of Missclassification (Costs) to Discriminate **Normally Distributed Population into TWO Groups**. 
for more you can read book 'Applied Multivariate Statistical Analysis by Richard Johnson an Dean Wichern'

###### How to Import:
**from QDA import qda**    
- place QDA.py in directory of your code
- Make sure to preinstall numpy,pandas,scipy,sklearn 

###### Usage:
## To check Normality of Multivariate data
**qda.check_norm(X,names=False)**         
- #X : your dataframe
- #names : True will annoniate Observation Numbers on final plot
- Output= Method will return Number of Observation below QQ line and plot of chisquare scores

## To fit QDA to Training data
**qda.fit(y,X,c12=1,c21=1,plot=True,obs=0,names=False)**
- #y: vector of  labels of  train data (Usually 0 and 1)
- #X: dataframe of features
- #plot: True will plot all Scores with Ideal Score
- #c12: Cost of Missclassification if Observation belongs to second group but classified into first group
- #c21: Cost of Missclassification if Observation belongs to First group but classified into second group
- #obs: integered valued; denotes Number of Observations from first row to be included into plot while zero indicates that all of the observations are included in Plot
- #names: True will annoniate Observation Numbers on final plot
- Output=Method will return,
- 1.Weights assigns to each variable (helps to understand which variable is more important in Discrimination
- 2.Ideal Score from missclassification
- 3.Accuracy; Calculated by using all observations as training data
- 4.Confusion Matrix
- 5.Plot; where Red line denote Ideal Score 		
- Colors indicates True Grouping of data
- Position of Obsrvation indicates Calculated Grouping of data
	
## To Predict Grouping of new Observation
**qda.predict(x,y,X,c12=1,c21=1)**
- #x: new Observation under consideration
- #y: vector of  labels of  train data (Usually 0 and 1)
- #X: dataframe of features of training data
- #c12: Cost of Missclassification if Observation belongs to second group but classified into first group
- #c21: Cost of Missclassification if Observation belongs to First group but classified into  second group
- Output=Method will return score of new Observation which can be compared with Ideal score

## Known Issues:
- qda.fit() fails if Dispersion matrix is Singular
