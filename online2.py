
import scipy.io as spio
import numpy as np
import json
import matplotlib.pyplot as plt
import quadprog as qp
import cvxopt_ as cvx

# for the first 30 minutes wait for the data to come 
# in that time period that is going to be accumulated 
# by the variables. After that we can continue learning 
# . The code will then be doing two things. Predicting
# the future temperature and at the same time learning the model

''' 
d = sio.loadmat('data.mat')
data1 = d.data_temp
print(data1)
'''
def loadmat(filename):
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
   
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def load_input(filename):
	with open(filename) as f:
		data = json.load(f)
	return data


def get_sum(lst,step):
	# print np.shape(lst)
	# print(lst)
	sm_lt = []
	count = 0
	l = len(lst)
	while(1):
		# sm_lt = np.append(sm_lt,np.sum(lst[count:count+step]))
		sm_lt = np.append(sm_lt,np.sum(lst[count:count+step]))
		count = count + step
		if count >= l:
			break
	return np.reshape(sm_lt,[1,len(sm_lt)])


def train_model(labels,feature,number_sensors):
	sh = np.shape(feature)
	col = sh[1]
	# print(feature)
	f = np.hstack((np.array(feature[:,1:2]),np.array(feature[:,number_sensors:col])))
	print("output dowen")
	# print f

	for i in range(number_sensors) : 

		f = np.hstack((np.array(feature[:,i:i+1]),np.array(feature[:,number_sensors:col])))
		l = labels[:,i:i+1]
		trans = np.transpose(f)
		m = np.matmul(trans,f)
		pinv = np.linalg.pinv(m,rcond = 1e-10)
		theeta = np.matmul(np.matmul(pinv,trans),l)
		if i == 0:
			Big_theeta = theeta
		else:
			Big_theeta = np.hstack((Big_theeta,theeta)) 
	return Big_theeta


	#return theeta


def my_quadprog(Bigtheeta,row_theeta,col_theeta,number_sensors):

	Big_theeta = np.transpose(Bigtheeta)
	row = col_theeta
	col = row_theeta
	W_o = Big_theeta[:,col-1:col]
	A = Big_theeta[:,0:1]
	# print np.shape(A)
	A_y = np.diag(A[:,0])
	# print A_y
	print np.shape(A_y)
	print np.shape(Big_theeta)
	Weight_u = Big_theeta[:,1:col-1]
	print np.shape(Weight_u)
	H = np.identity(col-2)
	y_des = 24.0*np.ones((number_sensors,1))
	y_init = 20.9*np.ones((number_sensors,1))
	beq = y_des - np.matmul(A_y,y_init) -W_o
	Aeq = Weight_u;
	A = np.vstack((Aeq,-Aeq))
	A = -A
	print col
	A = np.vstack((A,np.identity(col-2)))	
	
	A = np.vstack((A,-1*np.identity(col-2)))
	delta = 2
	b = np.vstack((beq+delta,-beq-delta))
	b=-b
	b = np.vstack((b,150.0*np.ones((col-2,1))))
	b = np.vstack((b,500*np.ones((col-2,1))))

	print np.shape(b)
	qp.solve_qp(H,np.zeros((col-2,1)),np.transpose(A),b,0)


def my_cvxopt(Bigtheeta,row_theeta,col_theeta,number_sensors):
	print "inside"
	Big_theeta = np.transpose(Bigtheeta)
	row = col_theeta
	col = row_theeta
	W_o = Big_theeta[:,col-1:col]
	A = Big_theeta[:,0:1]
	# print np.shape(A)
	A_y = np.diag(A[:,0])
	# print A_y
	print np.shape(A_y)
	print np.shape(Big_theeta)
	Weight_u = Big_theeta[:,1:col-1]
	print np.shape(Weight_u)
	H = np.identity(col-2)
	y_des = 24.0*np.ones((number_sensors,1))
	y_init = 20.9*np.ones((number_sensors,1))
	beq = y_des - np.matmul(A_y,y_init) -W_o
	Aeq = Weight_u;
	A = np.vstack((Aeq,-Aeq))
	print col
	A = np.vstack((A,-1*np.identity(col-2)))	
	
	A = np.vstack((A,np.identity(col-2)))
	delta = 2
	b = np.vstack((beq+delta,-beq-delta))
	b = np.vstack((b,-150.0*np.ones((col-2,1))))
	b = np.vstack((b,500.0*np.ones((col-2,1))))
	out = cvx.cvxopt_solve_qp(H,np.zeros((col-2,1)),A,b)
	return out




def test_model(feature,number_sensors,labels,sensor,theeta):
	sh = np.shape(feature)
	col = sh[1]
	print(sh)
	test = np.hstack((np.array(feature[600:800,sensor:sensor+1]),np.array(feature[600:800,number_sensors:col])))
	Y = labels[600:800,sensor:sensor+1]
	print np.shape(test)
	print np.shape(theeta[:,1])
	Y_predicted = np.matmul(test,theeta[:,sensor:sensor+1])

	print np.shape(Y_predicted)
	error = np.sqrt(np.matmul(np.transpose(Y_predicted - Y),Y_predicted-Y))
	abs_error  = np.mean(np.absolute(Y_predicted-Y))
	print abs_error
	print error
	print(np.mean(Y))
	print np.mean(Y_predicted)
	# plt.plot(range(197),Y_predicted,range(197),Y)
	# plt.show()


	# print real_time_stack[1] #np.shape(real_time_stack)
		
	#'''

#get the size of data

# declare the prediction in fue




