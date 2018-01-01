import xlrd
import pprint
import numpy as np
import math
import copy
import os
# 计算两个点之间的距离值
def distance_xy(x1,x2,y1,y2):
	return math.sqrt((111 * (x1-x2))**2+(102*(y1-y2))**2)
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
def read_xls_b(dir_path,txt_dir):
	data_k = read_txt(txt_dir)
	data_a = readxls(dir_path,"Sheet1")
	del data_a[0]
	data_all = []
	#pprint.pprint(data_a)
	for k in range(len(data_a)):
		Temp = [float(data_a[k][1].split()[0]),float(data_a[k][1].split()[1]),data_a[k][2],data_k[k],data_a[k][4]]
		data_all.append(Temp)
	return np.array(data_all)
# 将数据分为若干类,按照价格分类
def seperate_data(data_all,data_all_a):
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
			task_all_a.append([numbers,distance_a/numbers,value_all/1878*835,time_aver/numbers/60,value_aver_credit/numbers/680])
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
			#print(value)
			if  value < 3:
				distance_a += value
				numbers += 1
		#print(numbers)
		if numbers != 0:
			task_all_b.append([numbers,distance_a/numbers])
		else:
			task_all_b.append([0,0])
	return np.array(task_all_b)
#读取文件中的时间点时刻
def read_txt(dir_file):
	data_k = []
	fp = open(dir_file)
	while True:
		line = fp.readline()
		if not line:
			break
		line = line.strip().split(":")
		hours = float(line[0])
		mintine = float(line[1])
		line = hours * 60 + mintine - (6*60 + 30)
		data_k.append(line)
	fp.close()
	return data_k
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
def Main():
	dir_path_a = "C:\\Users\\TTfg\\Desktop\\国赛文件\\CUMCM2017Problems\\B\\附件一：已结束项目任务数据.xls"
	dir_path_b = "C:\\Users\\TTfg\\Desktop\\国赛文件\\CUMCM2017Problems\\B\\附件二：会员信息数据.xlsx"
	dir_file = "C:\\Users\\TTfg\\Desktop\\国赛文件\\NewFlies\\time.txt"
	sheet_name_b = "Sheet1"
	sheet_name_a = "t_tasklaunch"
	#读取文件一、二，并将数据保存到两个变量之中
	data_all_a = read_xls_a(dir_path_a)
	data_all_b = read_xls_b(dir_path_b,dir_file)
	# 计算距离值物与会员之间的距离,以及平均距离，平均限额的表示
	task_all_a = calculate_dis_extra(data_all_a,data_all_b)
	# 计算物与物之间的个数值
	task_all_b = calculate_num(data_all_a)
	#将数据分为若干类
	task_all = np.hstack((task_all_a,task_all_b))
	#del task_all_a
	del task_all_b
	save_file(task_all,"data.txt")
	#---------------------------------------------------------------
	#分类问题，分为五种类型
	Arrs_all = seperate_data(task_all,data_all_a)
	# 保存文件
	dir_path = "Datas"
	save_flie_a(Arrs_all,dir_path)
	#---------------------------------------------------------------
if __name__ == "__main__":
	Main()