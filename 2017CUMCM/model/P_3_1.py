import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xlrd
#读取XLS表格数据
def readxls(dir_path,sheet_name):
	workbook = xlrd.open_workbook(dir_path)
	booksheet = workbook.sheet_by_name(sheet_name)#
	all_data = []
	for k in range(booksheet.nrows):
		Temp = booksheet.row_values(k)
		all_data.append(Temp)
	return all_data
#主函数
def train_Q():
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
	#----------------------------------------------
	#将30个类别的数据，每个中心进行标价平均处理
	Length = len(xy)
	price_all = np.zeros(Length,dtype = 'float64')
	price_num = np.zeros(Length,dtype = 'int')
	for k in range(len(datas_y)):
		price_all[y_pred[k]] += datas_y[k][3]
		price_num[y_pred[k]] += 1
	for k in range(Length):
		price_all[k] /= price_num[k]
	#----------------------------------
	#将数据中心点以及任务标价展示到图中
	plt.scatter(x,y,c = y_pred)
	col = [0,1,0,1]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='r', markersize=8)
	for k in range(len(xy)):
		plt.text(xy[k][0],xy[k][1],("(%0.2f,%0.2f)"%(xy[k][0],xy[k][1])))
	plt.title("Data of things distribution schematic")
	plt.ylabel("Longtitude")
	plt.xlabel("Lantitude")
	plt.show()
	# ---------------------------------------------
	#读取文件二中的数据
	dir_path_q = "附件二：会员信息数据.xlsx"
	datas_y_q = readxls(dir_path_q,"Sheet1")
	del datas_y_q[0]
	#------------------------------------
	#用户标号
	labels_all_b = []
	for k in range(len(datas_y_q)):
		labels_all_b.append(datas_y_q[k][0])
	#------------------------------------
	#读取文件二用户的其他属性的读取
	data_longtitude_lantitude_q = []
	data_value = []
	data_credit = []
	data_time = []
	for k in range(len(datas_y_q)):
		Temp = datas_y_q[k][1].split()
		Temp = [float(Temp[0]),float(Temp[1])]
		data_value.append(datas_y_q[k][2])
		data_longtitude_lantitude_q.append(Temp)
		data_credit.append(datas_y_q[k][4])
		data_time.append(float("%0.2f"%(datas_y_q[k][3]*24 - 6.5)))
		#print(float("%0.2f"%(datas_y_q[k][3]*24 - 6.5)))
	data_longtitude_lantitude_q = np.array(data_longtitude_lantitude_q)
	data_value = np.array(data_value)
	data_credit = np.array(data_credit)
	data_time = np.array(data_time)
	predict_val = ModelKm.fit_predict(data_longtitude_lantitude_q)
	# ---------------------------------------------
	#将用户的位置放入到训练好的KMeans模型中进行分析处理，并标在图中，包含分类的中心点
	x = data_longtitude_lantitude_q[:,0]
	y = data_longtitude_lantitude_q[:,1]
	plt.figure()
	plt.scatter(x,y,c = predict_val)
	col = [0,1,0,1]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='r', markersize=8)
	for k in range(len(xy)):
		plt.text(xy[k][0],xy[k][1],("(%0.2f,%0.2f)"%(xy[k][0],xy[k][1])))
	plt.title("Data of people distribution schematic")
	plt.ylabel("Longtitude")
	plt.xlabel("Lantitude")
	plt.show()
	# ---------------------------------------------
	#将用户的若干个属性进行平均化处理(配额,时间处理，信誉度平均处理)
	Length = len(xy)
	extra_all = np.zeros(Length,dtype = 'float64')
	price_num = np.zeros(Length,dtype = 'int')
	credit_all = np.zeros(Length,dtype = 'float64')
	time_all = np.zeros(Length,dtype = 'float64')
	for k in range(len(predict_val)):
		extra_all[predict_val[k]] += data_value[k]
		credit_all[predict_val[k]] += data_credit[k]
		time_all[predict_val[k]] += data_time[k]
		price_num[predict_val[k]] += 1
	for k in range(Length):
		extra_all[k] /= price_num[k]
		credit_all[k] /= price_num[k]
		time_all[k] /= price_num[k]
	#------------------------------------------------
	#写入文件处理
	fp_p =open("fileA.csv","w")
	line = "任务号码,任务gps纬度,任务gps经度,任务标价,任务执行情况\n"
	fp_p.write(line)
	for k in range(len(y_pred)):
		line = labels_all[k][0] + "," + str(data_longtitude_lantitude_a[k][0]) + "," + str(data_longtitude_lantitude_a[k][1])+","
		fp_p.write(line)
		line = str(price_all[y_pred[k]]) + "," +str(labels_all[k][1]) + "\n"
		fp_p.write(line)
	fp_p.close()
	# -----------------------------------------------
	fp_p =open("fileB.csv","w")
	line = "会员编号,会员位置(GPS),预订任务限额,预订任务开始时间,信誉值\n"
	fp_p.write(line)
	for k in range(len(predict_val)):
		line = datas_y_q[k][0] + "," + str(data_longtitude_lantitude_q[k][0]) + " " + str(data_longtitude_lantitude_q[k][1])+","
		fp_p.write(line)
		line = str(extra_all[predict_val[k]]) + "," +str(time_all[predict_val[k]]) + "," +str(credit_all[predict_val[k]]) + "\n"
		fp_p.write(line)
	fp_p.close()
	# ----------------------------------------------------
if __name__ == "__main__":
	train_Q()
