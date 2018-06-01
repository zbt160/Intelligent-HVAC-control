from HVACLib import HVAC
import time
import requests
import json
import random
import online2 as lr

thingSpeakURL = 'http://192.168.0.2:5000/'

myhvac = HVAC()

datalogging  = False
# desired1 = eval(input("Enter desired1 temp (in celcius): "))
# desired2 = eval(input("Enter desired2 temp (in celcius): "))
# desired3 = eval(input("Enter desired2 temp (in celcius): "))

myinput = input("Enter \"now\" to start experiment")
myhvac.reset()
time.sleep(30*60)


if myinput == "now":
  init_time = time.time()
else: 
  init_time = time.time()
  myinput = input("Enter \"now\" to start experiment")

temp_meas = [[] for i in range(5)]
valve_meas = [[] for i in range(4)]
co2_meas = []
RH_meas = []
timestamp = []
U = [[] for i in range(4)]
indx = 0

count = 0

while True: 
	start_time = time.time()
	timestamp.append(myhvac.getTime(datalogging))

	# Set point change
	# if (indx % 30 == 0): 
	#     if (x == 1):
	#         myhvac.setTemp(desired1)
	#         x = 2
	#         sp = desired1
	#     elif (x == 2):
	#         myhvac.setTemp(desired2)
	#         x = 3
	#         sp = desired2
	#     else:
	#         myhvac.setTemp(desired3)
	#         x = 1
	#         sp = desired3
	# setpoint.append(sp)

	# Input_level change
	if indx % 5 == 0:
		randnum = random.randrange(30, 100, 5) #choose a value between 30% and myrange in steps of 5%
		print(randnum)
		myhvac.setEP(1, randnum)
		U[0].append(randnum)
		randnum = random.randrange(30, 100, 5) #choose a value between 30% and myrange in steps of 5%
		print(randnum)
		myhvac.setEP(2, randnum)
		U[1].append(randnum)
		randnum = random.randrange(30, 100, 5) #choose a value between 30% and myrange in steps of 5%
		print(randnum)
		myhvac.setEP(3, randnum)
		U[2].append(randnum)
		randnum = random.randrange(30, 100, 5) #choose a value between 30% and myrange in steps of 5%
		print(randnum)
		myhvac.setEP(4, randnum)
		U[3].append(randnum)
		myhvac.setFan("medium")

	# HVAC sensors measurement
	for i in range(5):
		temp_meas[i].append(myhvac.getTemp(i+1, datalogging))


	if count > 0:
		real_data = np.reshape(np.append(temp_meas,[U[0],U[1],U[2],U[3]])
		real_time_stack = np.append(real_time_stack,real_data,axis = 0)				
		if count >= future_time:
			list_sum_ahu_heat = lr.get_sum(real_time_stack[count -future_time:count,1],step)
			list_sum_ahu_cool = lr.get_sum(real_time_stack[count -future_time:count,2],step)
			list_sum_rad1 = lr.get_sum(real_time_stack[count -future_time:count,3],step)
			list_sum_rad2 = lr.get_sum(real_time_stack[count -future_time:count,4],step)
			feature_temp = np.reshape(np.append(real_time_stack[count-future_time,0:number_sensors],np.append([list_sum_ahu_heat,list_sum_ahu_cool,list_sum_rad1,list_sum_rad2],[1])),[1,1+number_sensors+(future_time/step)*4])
			labels_temp = np.reshape(real_time_stack[count,0:number_sensors],[1,number_sensors])
			if count == future_time:
				feature = feature_temp
				labels = labels_temp				
			else:
				feature = np.append(feature,feature_temp,axis = 0)
				labels = np.append(labels,labels_temp,axis = 0)
			#print feature
			if count > 600 :
				theeta = lr.train_model(labels,feature,number_sensors)
				spio.savemat('mydata.mat', mdict={'theeta': theeta})
				shp = np.shape(theeta)
				u=lr.my_cvxopt(theeta,shp[0],shp[1],number_sensors) #my_quadprog(theeta,shp[0],shp[1],number_sensors)
				print u
			# if count == 800:
			# 	test_model(feature,number_sensors,labels,1,theeta)


		count = count + 1

	
	else:
		real_data = np.append(temp_meas,[U[0],U[1],U[2],U[3]])
		len_real_data = len(real_data)
		real_data = np.reshape(real_data,[1,len_real_data])
		# print real_data
		count = 1
		real_time_stack = real_data #np.reshape(real_data,[1,len(real_data)])
		labels = np.array([])
		feature = np.array([])
		future_time = 30
		step = 5
		number_sensors = 5






	co2_meas.append(myhvac.getCO2(datalogging))
	RH_meas.append(myhvac.getHumid(datalogging))



	## HVAC valve opening levels measurement 
	for i in range(4):
		valve_meas[i].append(myhvac.getEPlevel(i+1, datalogging))  

	# pv = sum([collumn[-1] for collumn in temp_meas])/5
	# myhvac.setFeedbackTemp(pv)

	if indx == 0:
		init_temp = sum([collumn[0] for collumn in temp_meas])/5

	# per_error = (sp-pv)/sp
	# error.append(per_error)   

	time.sleep(34) # Collecting data in every 1 minute
	end_time = time.time()
	print(end_time-start_time)

	print(indx)
	# Data Logging
	if indx % 60 == 0:
		data =  {'C'  : co2_meas,
				 'T1' : temp_meas[0],
				 'T2' : temp_meas[1],
				 'T3' : temp_meas[2],
				 'T4' : temp_meas[3],
				 'T5' : temp_meas[4],
				 'H'  : RH_meas,
				 'North_R' :  valve_meas[0],
				 'East_R'  :  valve_meas[1],
				 'AC_heat' :  valve_meas[2],
				 'AC_cool' :  valve_meas[3],
				 'Time_stamp': timestamp,
				 'Initial Temp':init_temp,
				 'Input': U
				 }
		with open(str(time.strftime("%M_%H_%d_%m"))+'_data_U_Zaid.json', 'w') as fp: # save data in a file
			json.dump(data, fp, indent = 4, sort_keys = True,
							  separators = (',', ': '), ensure_ascii = False)
		print(data)
	indx = indx + 1 

end_time = time.time()
print(end_time - init_time)

myhvac.reset()
myhvac.close()
