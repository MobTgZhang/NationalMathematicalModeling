dir_a = 'C:\\Users\\TTfg\\Desktop\\国赛文件\\a.txt';%物品的经纬度位置表示
dir_b= 'C:\\Users\\TTfg\\Desktop\\国赛文件\\b.txt';%人的位置的经纬度表示
dir_value = 'C:\\Users\\TTfg\\Desktop\\国赛文件\\value.txt';%价格
dir_task = 'C:\\Users\\TTfg\\Desktop\\国赛文件\\task.txt';%任务完成的基本情况
dir_rank = 'C:\\Users\\TTfg\\Desktop\\国赛文件\\people_rank.txt';%人物的信誉度处理
dir_value_tt = 'C:\\Users\\TTfg\\Desktop\\国赛文件\\value_ttt.txt';%人的配额分配
ThingsData= load(dir_a);%物品的经纬度位置表示
ClientData = load(dir_b);%人的位置的经纬度表示
ValueData = load(dir_value);%价格
taskValue = load(dir_task);%任务完成的基本情况
rankValue = load(dir_rank);%人物的信誉度处理
Extra_Value = load(dir_value_tt);%人的配额分配
x = ThingsData(:,1);
y = ThingsData(:,2);
z = ValueData;
str_all = zeros(length(x),1);
for k =1:length(ClientData)
    str_c = People_rank(rankValue(k));
    str_c = strcat(str_c,'.');
    Xis = ClientData(k,1);
    Yis = ClientData(k,2);
    plot(Xis,Yis,str_c,'MarkerSize',20);
    hold on
end
for k = 1:length(x)
    str_c = Color_defination(z(k));
    str_c = strcat('+',str_c);
    plot(x(k),y(k),str_c,'MarkerSize',10);
    hold on
end