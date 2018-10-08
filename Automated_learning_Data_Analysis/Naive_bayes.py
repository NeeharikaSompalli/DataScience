import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

def load_data():
		train_data = pd.read_csv("hw2q5.csv",
		dtype = None,
		index_col = ["Id"],
		delimiter= ',', 
		skiprows=1, 
		names=["Id","patient age", "spectacle prescription", "astigmatic", "tear production rate", "Class"])
		
		train_data["patient_age_encoded"] = LabelEncoder().fit_transform(train_data["patient age"])
		train_data["spectacle_prescription_encoded"] = LabelEncoder().fit_transform(train_data["spectacle prescription"])
		train_data["astigmatic_encoded"] = LabelEncoder().fit_transform(train_data["astigmatic"])
		train_data["tear_production_rate_encoded"] = LabelEncoder().fit_transform(train_data["tear production rate"])
		train_data["Class_encoded"] = LabelEncoder().fit_transform(train_data["Class"])
		
		train_X = train_data[["patient_age_encoded", "spectacle_prescription_encoded", "astigmatic_encoded", "tear_production_rate_encoded"]]
		
		train_Y = train_data[["Class_encoded"]]
		
		print(train_data.head(25))
		#print(train_X)
		return train_X, train_Y

def cross_validation(train_X, train_Y):
		
		for I in range(1,6):
			train_set = train_X.copy()
			train_set_Y = train_Y.copy()
			test_set = train_X.copy()
			test_set_Y = train_Y.copy()
			for i in range(1,25):
				if(i%5 + 1 == I):
					train_set.drop(i, inplace=True)
					train_set_Y.drop(i,inplace=True)
			test_set = test_set[~test_set.isin(train_set).all(1)]
			test_set_Y = test_set_Y[~test_set_Y.isin(train_set_Y).all(1)]
			nb(train_set,train_set_Y.values.ravel(), test_set, test_set_Y.values.ravel())
			dt(train_set,train_set_Y.values.ravel(), test_set, test_set_Y.values.ravel())			
			

def nb(train_X, train_Y, test_set, test_set_Y):
		print('Naive Bayes')
		print('------------------------------------------------------------------')
		print('Test data')
		print(test_set)
		print('True class: {0}'.format(test_set_Y))
		clf = GaussianNB()
		#clf = MultinomialNB()
		#clf = BernoulliNB()
		clf.fit(train_X, train_Y)
		predictions = clf.predict(test_set)
		print('Predicted class: {0}'.format(predictions))
		print('Accuracy: {0}'.format(accuracy_score(predictions,test_set_Y)))
		print(' ')

def dt(train_X, train_Y, test_set, test_set_Y):
		print('Decision Tree')
		print('------------------------------------------------------------------')
		print('Test data')
		print(test_set)
		print('True class: {0}'.format(test_set_Y))
		clf2 = tree.DecisionTreeClassifier()
		clf2.fit(train_X, train_Y)
		predictions = clf2.predict(test_set)
		print('Predicted class: {0}'.format(predictions))
		print('Accuracy: {0}'.format(accuracy_score(predictions,test_set_Y)))
		print(' ')
		
def main():
	train_X,train_Y = load_data()
	cross_validation(train_X, train_Y)
	
if __name__== "__main__":
  main()