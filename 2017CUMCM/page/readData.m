dir_a = 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\a.txt';%��Ʒ�ľ�γ��λ�ñ�ʾ
dir_b= 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\b.txt';%�˵�λ�õľ�γ�ȱ�ʾ
dir_value = 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\value.txt';%�۸�
dir_task = 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\task.txt';%������ɵĻ������
dir_rank = 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\people_rank.txt';%����������ȴ���
dir_value_tt = 'C:\\Users\\TTfg\\Desktop\\�����ļ�\\value_ttt.txt';%�˵�������
ThingsData= load(dir_a);%��Ʒ�ľ�γ��λ�ñ�ʾ
ClientData = load(dir_b);%�˵�λ�õľ�γ�ȱ�ʾ
ValueData = load(dir_value);%�۸�
taskValue = load(dir_task);%������ɵĻ������
rankValue = load(dir_rank);%����������ȴ���
Extra_Value = load(dir_value_tt);%�˵�������
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