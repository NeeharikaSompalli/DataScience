import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = [[27,6,-1], [-6,2,-1], [2,2,1], [36,2,1], [-8,4,-1], [40,2,1], [35,4,-1], [30,2,1], [20,6,1], [-1,4,-1]]

data_wo_class = [[27,6], [-6,2], [2,2], [36,2], [-8,4], [40,2], [35,4], [30,2], [20,6], [-1,4]]

actual_class = [-1,-1,1,1,-1,1,-1,1,1,-1]

def euclidianDis(x1, x2):
	return round(math.sqrt(((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)), 2)

def parta():
	result5 = []
	result10 = []
	for x in data:
		result5.append(euclidianDis(data[4], x))
		result10.append(euclidianDis(data[9], x))
	
	result5.pop(4);
	result10.pop(9);
	
	print('Euclidian distance for all other data points from (-8,4)')
	print(result5)
	print(' ')
	print('Euclidian distance for all other data points from (-1,4)')
	print(result10)
	print(' ')
	
def partb():
	
	result = []
	min_list = []
	min_index = []
	percent_result = []
	predicted_result = [] 
	
	for x in data:
		set = []
		for y in data:	
			set.append(euclidianDis(x, y))
		result.append(set)

	print('Euclidian distance from every data point to every other data point')	
	print(result)
	print(' ')
	
	for i,row in enumerate(result):
		row[i] = 999999999
		min_list.append(min(row));
		min_index.append(np.argmin(row));
	
	print('Minimum distance for each data point')
	print(min_list)
	print(' ')
	print('Nearest data point index(ID - 1) for each data point')
	print(min_index)
	print(' ')
	
	for i,point in enumerate(data):
		if(point[2] == data[min_index[i]][2]):
			percent_result.append(100)
			predicted_result.append(data[min_index[i]][2])
		else:
			percent_result.append(0)
			predicted_result.append(data[min_index[i]][2])
	
	print('True class: (-1 is negative and 1 is positive)')
	print('[-1, -1, 1, 1, -1, 1, -1, 1, 1, -1]')
	print(' ')
	print('Predicted class: (-1 is negative and 1 is positive)')
	print(predicted_result)
	print(' ')
	print('Comapring the true class and predicted class(100 for correct and 0 for wrong)')
	print(percent_result)
	print(' ')
	print('Correct prediction is ')
	print(sum(percent_result) / float(len(percent_result)))	
	print(' ')

def partc():
	training = []
	training_actual = []
	test = []
	test_actual = []
	true_class = []
	predicted_class = []
	for I in range(1,6):
		for i in range(1,11):
			if(i%5 + 1 != I):
				training.append(data_wo_class[i-1])
				training_actual.append(actual_class[i-1])
			else:
				test.append(data_wo_class[i-1])
				test_actual.append(actual_class[i-1])
				
		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(training, training_actual)
		
		print(test)
		print('{0}, Actual class: {1}, Predicted class: {2}'.format(test[0],test_actual[0], neigh.predict([test[0]])))
		print('{0}, Actual class: {1}, Predicted class: {2}'.format(test[1],test_actual[1], neigh.predict([test[1]])))
		print(' ')
		
		true_class.append(test_actual[0])
		true_class.append(test_actual[1])
		
		predicted_class.append(neigh.predict([test[0]]))
		predicted_class.append(neigh.predict([test[1]]))
		
		training = []
		training_actual = []
		test = []
		test_actual = []
	
	print('True class: (-1 is negative and 1 is positive)')
	print(true_class)
	print('Predicted class: (-1 is negative and 1 is positive)')
	print(predicted_class)
	
	
def main():
	print('Part A')
	print('--------------------------------------------------------------------')
	parta()
	print('Part B')
	print('--------------------------------------------------------------------')
	partb()
	print('Part C')
	print('--------------------------------------------------------------------')
	partc()
	
if __name__== "__main__":
  main()