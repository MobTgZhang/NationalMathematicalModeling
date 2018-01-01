import math
import xlrd
import numpy as np
import os
import copy
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
#模型的基本定义
class BPNetWork(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super(BPNetWork,self).__init__()
		self.hidden = nn.Linear(input_dim,hidden_dim)
		self.output = nn.Linear(hidden_dim,output_dim)
	def forward(self,input_s):
		x = F.relu(self.hidden(input_s))
		x = self.output(x)
		return x
#输出为softmax函数分类器
class BPSoftMax(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super(BPSoftMax,self).__init__()
		self.hidden = nn.Linear(input_dim,hidden_dim)
		self.output = nn.Linear(hidden_dim,output_dim)
	def forward(self,input_s):
		x = F.relu(self.hidden(input_s))
		x = self.output(x)
		return F.softmax(x)
# 计算两个点之间的距离值
def distance_xy(x1,x2,y1,y2):
	return math.sqrt((111* (x1-x2))**2+(102*(y1-y2))**2)
#初始化数据
def initialize_data(Arrs):
	for k in range(len(Arrs[0])):
		Arrs[:,k] = (Arrs[:,k] - np.mean(Arrs[:,k]))/(np.std(Arrs[:,k]))
	return Arrs
# 读取xls文件
def readxls(dir_path,sheet_name):
	workbook = xlrd.open_workbook(dir_path)
	booksheet = workbook.sheet_by_name(sheet_name)#
	all_data = []
	for k in range(booksheet.nrows):
		Temp = booksheet.row_values(k)
		all_data.append(Temp)
	return all_data
# 将表一进行分类计算结果
def read_xls_a(dir_path):
	data_a = readxls(dir_path,"t_tasklaunch")
	del data_a[0]
	data_all = []
	for k in range(len(data_a)):
		data_all.append([data_a[k][1],data_a[k][2],data_a[k][3],data_a[k][4]])
	return np.array(data_all)
# 将表二进行分类计算结果
def read_xls_b(dir_path):
	data_a = readxls(dir_path,"Sheet1")
	del data_a[0]
	data_all = []
	#pprint.pprint(data_a)
	for k in range(len(data_a)):
		Temp = [float(data_a[k][1].split()[0]),float(data_a[k][1].split()[1]),data_a[k][2],(data_a[k][3]*24 - 6.5)*60,data_a[k][4]]
		data_all.append(Temp)
	return np.array(data_all)
# 将表三进行分类计算结果
def read_xls_c(dir_path):
	data_a = readxls(dir_path,"t_tasklaunch")
	del data_a[0]
	data_all = []
	for k in range(len(data_a)):
		data_all.append([data_a[k][1],data_a[k][2]])
	return np.array(data_all)
# 将数据分为若干类,按照价格分类
def seperate_data_AA(data_all,data_all_a):
	Length = len(data_all)
	T_a = []
	T_b = []
	T_e = []
	for k in range(Length):
		Temp = copy.deepcopy(data_all[k])
		Length = len(data_all_a[k])
		value = data_all_a[k][Length-2]
		if value < 70:
			T_a.append([k+1] + [data_all_a[k][Length-1],data_all_a[k][Length-2]] +Temp.tolist())
		elif value < 75:
			T_b.append([k+1] + [data_all_a[k][Length-1],data_all_a[k][Length-2]] + Temp.tolist())

		else:
			T_e.append([k+1] + [data_all_a[k][Length-1],data_all_a[k][Length-2]] + Temp.tolist())
	Arrs = []
	Arrs.append(T_a)
	Arrs.append(T_b)
	Arrs.append(T_e)
	del T_a
	del T_b
	del T_e
	return np.array(Arrs)
# 将数据分为若干类,按照价格分类
def seperate_data(task_all,data_all_a):
	Length = len(task_all)
	Arrs = []
	taget_score = []
	for k in range(Length):
		if data_all_a[k][-1] == 1.0:
			Arrs.append(task_all[k].tolist())
			taget_score.append(data_all_a[k][2])
	return np.array(Arrs),np.array(taget_score)
# 计算距离值物与会员之间的距离,以及平均距离，平均限额的表示
def calculate_dis_extra(data_all_a,data_all_b):
	task_all_a = []
	for k in range(len(data_all_a)):
		numbers = 0
		distance_a = 0
		value_all = 0
		value_aver_credit = 0
		time_aver = 0
		for j in range(len(data_all_b)):
			value = distance_xy(data_all_a[k][0],data_all_b[j][0],data_all_a[k][1],data_all_b[j][1])
			if  value <3:
				value_all += data_all_b[j][2]
				value_aver_credit += data_all_b[j][4]
				time_aver += data_all_b[j][3]
				distance_a += value
				numbers += 1
		if numbers != 0:
			task_all_a.append([numbers,distance_a/numbers,value_all/1878/numbers*835,time_aver/numbers/60,value_aver_credit/numbers/680])
		else:
			task_all_a.append([0,0,0,0,0])
	return np.array(task_all_a)
# 计算物与物之间的个数值
def calculate_num(data_a):
	task_all_b = []
	for k in range(len(data_a)):
		numbers = -1
		distance_a = 0
		for j in range(len(data_a)):
			value = distance_xy(data_a[k][0],data_a[j][0],data_a[k][1],data_a[j][1])
			if  value < 2:
				distance_a += value
				numbers += 1
		if numbers != 0:
			task_all_b.append([numbers,distance_a/numbers])
		else:
			task_all_b.append([0,0])
	return np.array(task_all_b)
# 保存数据值
def save_file(task_all,dir_path):
	fp_r = open(dir_path,"w")
	for k in range(len(task_all)):
		for j in range(len(task_all[k])):
			fp_r.write(str(task_all[k][j]) + " ")
		fp_r.write("\n")
	fp_r.close()
# 保存分完类之后的文件
def save_flie_a(task_all,dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)
	#print(task_all.shape)
	Length = len(task_all)
	for k in range(Length):
		fp_rs = open(dir_path + "/data_" + str(k) + ".txt","w")
		for j in range(len(task_all[k])):
			for r in range(len(task_all[k][j])):
				fp_rs.write(str(task_all[k][j][r]) + " ")
			fp_rs.write("\n")
		fp_rs.close()
#主函数
def MainA():
	dir_path_a = "附件一：已结束项目任务数据.xls"
	dir_path_b = "附件二：会员信息数据.xlsx"
	# 读取文件一、二，并将数据保存到两个变量之中
	data_all_a = read_xls_a(dir_path_a)
	data_all_b = read_xls_b(dir_path_b,dir_file)
	# 计算距离值物与会员之间的距离,以及平均距离，平均限额的表示
	task_all_a = calculate_dis_extra(data_all_a,data_all_b)
	# 计算物与物之间的个数值
	task_all_b = calculate_num(data_all_a)
	#将数据分为若干类
	task_all = np.hstack((task_all_a,task_all_b))
	del task_all_b
	save_file(task_all,"data.txt")
	#---------------------------------------------------------------
	#分类问题，分为三种类型
	Arrs_all = seperate_data_AA(task_all,data_all_a)
	# 保存文件
	dir_path = "Datas"
	save_flie_a(Arrs_all,dir_path)
	#---------------------------------------------------------------
#读取训练集合
def Read_train_data():
	dir_path_a = "附件一：已结束项目任务数据.xls"
	dir_path_b = "附件二：会员信息数据.xlsx"
	#读取文件一、二，并将数据保存到两个变量之中
	data_all_a = read_xls_a(dir_path_a)
	data_all_b = read_xls_b(dir_path_b)
	# 计算距离值物与会员之间的距离,以及平均距离，平均限额的表示
	task_all_a = calculate_dis_extra(data_all_a,data_all_b)
	# 计算物与物之间的个数值
	task_all_b = calculate_num(data_all_a)
	#将数据分为若干类
	task_all = np.hstack((task_all_a,task_all_b))
	del data_all_b
	del task_all_a
	del task_all_b
	return task_all,data_all_a
#利用模型一进行价格训练
def train_data_model_price(model_name):
	input_dim = 7 #输入节点个数
	hidden_dim = 15 #隐藏层节点个数
	output_dim = 1 #输出节点个数
	learning_rate  = 1e-2 #学习率大小
	epoches = 300 #循环的次数
	BpModel = BPNetWork(input_dim,hidden_dim,output_dim) #创建BPNetWork类
	optimizer = optim.SGD(BpModel.parameters(),lr = learning_rate) #定义一个优化器
	loss_func = nn.MSELoss() #均方差损失函数

	task_all,data_all_a = Read_train_data()#读取数据
	Arrs,taget_score = seperate_data(task_all,data_all_a)
	taget_score = taget_score/85#归一化处理
	Arrs = initialize_data(Arrs)#标准化数据
	#定义一个数组，用来保存损失函数的值
	data_ttf = []
	for epoch in range(epoches):
		Length = len(Arrs)
		loss_data = 0
		for k in range(Length):
			input_s = Variable(torch.FloatTensor(Arrs[k])).unsqueeze(0)
			predict_score = BpModel(input_s)
			taget_s = Variable(torch.FloatTensor([taget_score[k]])).unsqueeze(0)
			loss = loss_func(predict_score,taget_s)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_data += loss.data[0]
		loss_data = loss_data/Length
		data_ttf.append(loss_data)
		print(loss_data)
	#---------------------------------------------------------
	# 保存数据值，损失变化值
	data_file = "loss_Bpmodel_Price.txt"
	fp_t = open(data_file,"w")
	for k in range(len(data_ttf)):
		fp_t.write(str(data_ttf[k]) + "\n")
	fp_t.close()
	#画出损失函数的图像
	#---------------------------------------------------------
	x = np.linspace(1,epoches,epoches)
	y = np.array(data_ttf)
	plt.title("The final reslut show......")
	plt.ylabel("The average loss")
	plt.xlabel("The number of epoches")
	plt.plot(x,y,'-')
	plt.show()
	plt.savefig('model_loss_A.png')
	torch.save(BpModel,model_name)
	print("The trained model completed!")
#定义一个完成情况预测的一个还函数
def train_data_model_complete(model_name):
	input_dim = 8 #输入节点个数
	hidden_dim = 15 #隐藏层节点的个数
	output_dim = 2 #输出节点的个数
	epoches = 600 #训练循环的次数
	learning_rate = 0.01 #学习率的大小
	BpModel = BPSoftMax(input_dim,hidden_dim,output_dim) #构建一个BPSoftMax类
	optimizer = optim.SGD(BpModel.parameters(),lr = learning_rate) #构建优化器函数，使用SGD算法
	loss_func = nn.MSELoss() #均方差损失函数

	task_all,data_all_a = Read_train_data() #读取基本数据值
	Temp = data_all_a[:,2]
	Temp = Temp / 85
	Temp = Temp.reshape(len(Temp),1)
	task_all = initialize_data(task_all)
	task_all = np.hstack((task_all,Temp))
	data_ttf = []
	#训练过程
	for epoch in range(epoches):
		Length = len(task_all)
		loss_data = 0
		for k in range(Length):
			input_s = Variable(torch.FloatTensor(task_all[k])).unsqueeze(0)
			predict_score = BpModel(input_s)
			taget_s = None
			if data_all_a[k][3] == 0.0:
				taget_s = [1,0]
			else:
				taget_s = [0,1]
			taget_s = Variable(torch.FloatTensor(taget_s))
			loss = loss_func(predict_score,taget_s)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_data += loss.data[0]
		loss_data = loss_data/Length
		data_ttf.append(loss_data)
		print(loss_data)
	#---------------------------------------------------------
	# 保存损失变化值
	data_file = "loss_Bpmodel_Complete.txt"
	fp_t = open(data_file,"w")
	for k in range(len(data_ttf)):
		fp_t.write(str(data_ttf[k]) + "\n")
	fp_t.close()
	#---------------------------------------------------------
	#画图的表示方法
	x = np.linspace(1,epoches,epoches)
	y = np.array(data_ttf)
	plt.title("The final reslut show......")
	plt.ylabel("The average loss")
	plt.xlabel("The number of epoches")
	plt.plot(x,y,'-')
	plt.show()
	plt.savefig('model_loss_B.png')
	torch.save(BpModel,model_name)
	print("The trained model completed!")
#读取文件二以及文件三中的内容
def read_predict_data():
	dir_path_c = "附件三：新项目任务数据.xls"
	dir_path_b = "附件二：会员信息数据.xlsx"
	data_all_a = read_xls_c(dir_path_c)
	# 将表二进行分类计算结果
	data_all_b = read_xls_b(dir_path_b)
	# 计算距离值物与会员之间的距离,以及平均距离，平均限额的表示
	task_all_a = calculate_dis_extra(data_all_a,data_all_b)
	# 计算物与物之间的个数值
	task_all_b = calculate_num(data_all_a)
	fp_d = open("Test_data.txt","w")
	task_all = np.hstack((task_all_a,task_all_b))
	del data_all_b
	del task_all_a
	del task_all_b
	return task_all
#读取文件
def readData(dir_path):
	datas_all = []
	fp_rq = open(dir_path)
	while True:
		line = fp_rq.readline()
		if not line:
			break
		datas_all.append(float(line.strip()))
	fp_rq.close()
	return datas_all
#主函数
def MainE():
	model_name_q = "BpModel.model"
	if not os.path.exists(model_name_q):
		train_data_model_price(model_name_q)
	task_all_q =read_predict_data()
	BpModel_q = torch.load(model_name_q)
	data_all_w_q =[]
	task_all_q = initialize_data(task_all_q)
	for k in range(len(task_all_q)):
		input_s = Variable(torch.FloatTensor(task_all_q[k]))
		Temp = BpModel_q(input_s.unsqueeze(0))
		data_all_w_q.append(Temp.data.numpy()[0][0]*85)
	fp_d = open("price_predict_data.txt","w")
	for k in range(len(data_all_w_q)):
		fp_d.write("%0.1f"%(data_all_w_q[k]) + "\n")
	fp_d.close()
	data_all_w_q = np.array(data_all_w_q)
	#------------------------------------------------------------

	file_path_dir = "price_predict_data.txt"
	datas_all = readData(file_path_dir)
	datas_all =np.array(datas_all)
	Temp = datas_all /85
	Temp = Temp.reshape(len(Temp),1)
	task_all =read_predict_data()
	task_all = initialize_data(task_all)
	task_all = np.hstack((task_all,Temp))
	model_name = "BpModelComplete.model"
	if not os.path.exists(model_name):
		train_data_model_complete(model_name)
	BpModel = torch.load(model_name)
	data_all_w =[]
	for k in range(len(task_all)):
		input_s = Variable(torch.FloatTensor(task_all[k]))
		Temp = BpModel(input_s.unsqueeze(0))
		data_all_w.append(Temp.squeeze().data.numpy().tolist())
	fp_d = open("complete_predict_data.txt","w")
	Numbers = 0
	data_all_w = np.array(data_all_w)
	#pprint.pprint(data_all_w,compact = True)
	for k in range(len(data_all_w)):
		if data_all_w[k][1] > 0.5:
			Numbers += 1
	print(Numbers)
	for k in range(len(data_all_w)):
		fp_d.write(str(1 if data_all_w[k][1]>0.5 else 0)+ "\n")
	fp_d.close()
	#------------------------------------------------------------------------
	data_all_w_d = np.array([1 if data_all_w[k][1]>0.5 else 0 for k in range(len(data_all_w))])
	data_all_w_d = data_all_w_d.reshape(len(data_all_w_d),1)
	data_all_w_q = data_all_w_q.reshape(len(data_all_w_q),1)
	data_all_w_t = np.hstack((data_all_w_d,data_all_w_q))
	task_all =read_predict_data()
	task_all = np.hstack((task_all,data_all_w_t))
	fp_d = open("all_data_statics.txt","w")
	for k in range(len(task_all)):
		for j in range(len(task_all[k])):
			fp_d.write(str(task_all[k][j])+ " ")
		fp_d.write("\n")
	fp_d.close()
	#----------------------------------------------------------
#训练模型
def load_train_predict_model(model_nameA,model_nameB):
	if not os.path.exists(model_nameA):
		train_data_model_price(model_nameA)
	if not os.path.exists(model_nameB):
		train_data_model_complete(model_nameB)
	modelA = torch.load(model_nameA)
	modelB = torch.load(model_nameB)
	task_all,data_all_a = Read_train_data()
	task_all_q = copy.deepcopy(task_all)
	data_all_w_q = []
	data_all_w_r = []
	task_all = initialize_data(task_all)
	for k in range(len(task_all)):
		input_s = Variable(torch.FloatTensor(task_all[k]))
		TempA = modelA(input_s.unsqueeze(0))
		data_all_w_q.append(TempA.data.numpy()[0][0]*85)
	datas_all =np.array(data_all_w_q)
	Temp = datas_all/85
	Temp = Temp.reshape(len(Temp),1)
	task_all = np.hstack((task_all,Temp))
	for k in range(len(task_all)):
		input_s = Variable(torch.FloatTensor(task_all[k]))
		TempB = modelB(input_s.unsqueeze(0))
		data_all_w_r.append(TempB.data.squeeze().numpy().tolist())
	# ---------------------------------------------------------------
	fp_file = open("trained_score.txt","w")
	for k in range(len(data_all_w_r)):
		for j in range(len(data_all_w_r[k])):
			fp_file.write(str(data_all_w_r[k][j]) + " ")
		fp_file.write("\n")
	fp_file.close()
	Numbers = 0
	for k in range(len(data_all_w_r)):
		if data_all_w_r[k][1]> 0.5:
			Numbers += 1
	print(Numbers)
	# ---------------------------------------------------------------
	data_all_w_r =np.array(data_all_w_r)
	Temp = data_all_w_r[:,1]
	y = pow((Temp - 0.5),2)/0.25
	x = np.linspace(1,len(y),len(y))
	plt.plot(x,y,'r.')
	plt.xlabel("All points")
	plt.ylabel("The reliability")
	plt.title("The Reliability")
	plt.show()
	print(sum(y)/len(y))
	#-----------------------------------------------------------
	data_all_r =[1 if data_all_w_r[k][1]>0.5 else 0 for k in range(len(data_all_w_r))]
	data_all_r = np.array(data_all_r)
	data_all_r = data_all_r.reshape(len(data_all_r),1)
	datas_all = datas_all.reshape(len(datas_all),1)
	data_all_above = np.hstack((task_all_q,data_all_r,datas_all))
	fp_dr = open("all_data_statics_ttex.txt","w")
	del task_all
	for k in range(len(data_all_above)):
		for j in range(len(data_all_above[k])):
			fp_dr.write(str(data_all_above[k][j]) + " ")
		fp_dr.write("\n")
	fp_dr.close()
#主函数
def MainF():
	model_nameA,model_nameB = "BpModel.model","BpModelComplete.model"
	load_train_predict_model(model_nameA,model_nameB)
	MainE()
if __name__ == "__main__":
	MainF()
