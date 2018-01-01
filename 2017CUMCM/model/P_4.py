import pprint
from sklearn.cluster import KMeans
from P_1_2_4 import *
def load_txt(dir_path):
	price = []
	fp_rd = open(dir_path,"r")
	while True:
		line = fp_rd.readline()
		if not line:
			break
		price.append(float(line.strip()))
	fp_rd.close()
	return np.array(price)
def Main():
	dir_path = "附件一：已结束项目任务数据.xls"
	datas_y = readxls(dir_path,"t_tasklaunch")
	del datas_y[0]
	# ----------------------------------------
	#添加用户标号值以及完成任务情况值
	labels_all = []
	for k in range(len(datas_y)):
		Temp =[datas_y[k][0],datas_y[k][-1]]
		labels_all.append(Temp)
	# ----------------------------------------
	#经纬度处理过程
	data_longtitude_lantitude_a= []
	for k in range(len(datas_y)):
		Temp = [datas_y[k][1],datas_y[k][2]]
		data_longtitude_lantitude_a.append(Temp)
	data_longtitude_lantitude_a = np.array(data_longtitude_lantitude_a)
	#----------------------------------
	#用Kmeans聚类算法将835个数据分为30类
	ModelKm = KMeans(n_clusters = 30)
	ModelKm.fit(data_longtitude_lantitude_a)
	x = data_longtitude_lantitude_a[:,0]
	y = data_longtitude_lantitude_a[:,1]
	y_pred = ModelKm.labels_#预测值的表示
	xy = ModelKm.cluster_centers_#中心点的表示
	#----------------------------------------
	data_all = readxls("/home/asus/国赛文件夹/附件三：新项目任务数据.xls","t_tasklaunch")
	del data_all[0]
	all_la_tu = []
	for k in range(len(data_all)):
		all_la_tu.append([float(data_all[k][1]),(data_all[k][2])])
	all_la_tu = np.array(all_la_tu)
	label_all = ModelKm.fit_predict(all_la_tu)
	#------------------------------------------------
	price_data = load_txt("price_predict_data.txt")
	Length = len(xy)
	price_all = np.zeros(Length,dtype = "float64")
	num_all = np.zeros(Length,dtype = "int")
	Length = len(label_all)
	for k in range(Length):
		price_all[label_all[k]] += price_data[k]
		num_all[label_all[k]] += 1
	for k in range(len(price_all)):
		price_all[k] /= num_all[k]
	New_price = []
	for k in range(Length):
		New_price.append(price_all[label_all[k]])
	New_price = np.array(New_price)
	Temp = New_price / 85
	Temp = Temp.reshape(len(Temp),1)
	task_all_q =read_predict_data()
	task_all_q = initialize_data(task_all_q)
	task_all_q = np.hstack((task_all_q,Temp))
	#---------------------------------------------------
	model_name = "BpModelComplete.model"
	if not os.path.exists(model_name):
		train_data_model_complete(model_name)
	BpModel = torch.load(model_name)
	data_all_w =[]
	for k in range(len(task_all_q)):
		input_s = Variable(torch.FloatTensor(task_all_q[k]))
		Temp = BpModel(input_s.unsqueeze(0))
		data_all_w.append(Temp.squeeze().data.numpy().tolist())
	#--------------------------------------------------------
	datas_a = [1 if data_all_w[k][1]>0.5 else 0 for k in range(len(data_all_w))]
	Numbers = 0
	fp_d = open("reslut.txt","w")
	for k in range(len(datas_a)):
		if datas_a[k] ==1:
			plt.scatter(all_la_tu[k][0],all_la_tu[k][1],c ='r')
			Numbers += 1
		else:
			plt.scatter(all_la_tu[k][0],all_la_tu[k][1],c ='b')
		fp_d.write(str(datas_a[k])+"\n")
	fp_d.close()
	percentage = Numbers/len(datas_a)*100
	plt.title("The Prediction of Taskes:(completed: %0.2f%%,Numbers: %d)"%(percentage,Numbers))
	plt.ylabel("Longtitude")
	plt.xlabel("Lantitude")
	plt.show()
if __name__ == "__main__":
	Main()
