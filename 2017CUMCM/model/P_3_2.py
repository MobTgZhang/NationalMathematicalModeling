import torch
import csv
import numpy as np
from P_1_2_4 import *
#读取文件一中的数据，并进行相应的处理操作
def read_data_a(dir_path):
	csv_reader = csv.reader(open(dir_path, encoding='utf-8'))
	arrs = [row for row in csv_reader]
	del arrs[0]
	My_Arr=[]
	for row in arrs:
		Temp = [float(row[1]),float(row[2]),float(row[3]),float(row[4])]
		My_Arr.append(Temp)
	del arrs
	del csv_reader
	return np.array(My_Arr)
#读取文件二中的内容，并进行相应的处理操作
def read_data_b(dir_path):
	csv_reader = csv.reader(open(dir_path, encoding='utf-8'))
	arrs = [row for row in csv_reader]
	del arrs[0]
	My_Arr = []
	for row in arrs:
		Dft = row[1].split()
		Temp = [float(Dft[0]),float(Dft[1]),float(row[2]),float(row[3]),float(row[4])]
		My_Arr.append(Temp)
	del arrs
	del csv_reader
	return np.array(My_Arr)
#读取CSV文件，并处理相应的数据文件
def Read_CSV_Data():
	dir_path_a = "/home/asus/国赛文件夹/fileA.csv"
	dir_path_b = "/home/asus/国赛文件夹/fileB.csv"
	data_all_a = read_data_a(dir_path_a)
	data_all_b = read_data_b(dir_path_b)
	task_all_a = calculate_dis_extra(data_all_a,data_all_b)
	# 计算物与物之间的个数值
	task_all_b = calculate_num(data_all_a)
	#将数据分为若干类
	task_all = np.hstack((task_all_a,task_all_b))
	del data_all_b
	del task_all_a
	del task_all_b
	return task_all,data_all_a
#主函数，训练模型
def Main():
	model_nameA,model_nameB = "BpModel.model","BpModelComplete.model"
	train_model(model_nameA,model_nameB)#训练模型
#模型训练函数
def train_model(model_nameA,model_nameB):
	if not os.path.exists(model_nameA):
		train_data_model_price(model_nameA)
	if not os.path.exists(model_nameB):
		train_data_model_complete(model_nameB)
	#加载模型
	modelA = torch.load(model_nameA)
	modelB = torch.load(model_nameB)
	task_all,data_all_a = Read_CSV_Data()
	task_all_q = copy.deepcopy(task_all)
	data_all_w_q = []
	data_all_w_r = []
	#初始化数据
	task_all = initialize_data(task_all)
	#预测数据值
	for k in range(len(task_all)):
		input_s = Variable(torch.FloatTensor(task_all[k]))
		TempA = modelA(input_s.unsqueeze(0))
		data_all_w_q.append(TempA.data.numpy()[0][0]*85)
	datas_all =np.array(data_all_w_q)
	Temp = datas_all/85
	Temp = Temp.reshape(len(Temp),1)
	task_all = np.hstack((task_all,Temp))
	#预测数据值
	for k in range(len(task_all)):
		input_s = Variable(torch.FloatTensor(task_all[k]))
		TempB = modelB(input_s.unsqueeze(0))
		data_all_w_r.append(TempB.data.squeeze().numpy().tolist())
	data_all_r =[1 if data_all_w_r[k][1]>0.5 else 0 for k in range(len(data_all_w_r))]
	Numbers = 0
	#将数据写入数据文件中
	fp_file_q = open("All_data_extr.txt","w")
	for k in range(len(data_all_r)):
		fp_file_q.write(str(data_all_r[k]) + "\n")
		if data_all_r[k] == 1:
			Numbers += 1
	fp_file_q.close()
	print(Numbers)
	return data_all_r
if __name__ == "__main__":
	Main()
