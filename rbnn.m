%Input Data
Hydro_Param = readtable('hydroparam.csv');
Hydro_Param = table2array(Hydro_Param);
Turbin_Param = Hydro_Param(:, 1:4);

%Data Division
B_tip = Hydro_Param(:,1);
B_70 = Hydro_Param(:,2);
A_tip = Hydro_Param(:,3);
A_70 = Hydro_Param(:,4);
Cp = Hydro_Param(:, 5);

%Transpose Data
Turbin_Param = transpose(Turbin_Param);
Cp = transpose(Cp);

%rbnn param
eg = 0.0001;
sc = 100;

%build model(rbnn)
net = newrb(Turbin_Param, Cp);

%random parameter
Rand_Turbin_Param = readtable('RandParam.csv');
Rand_Turbin_Param = table2array(Rand_Turbin_Param);
Rand_Turbin_Param = transpose(Rand_Turbin_Param);

%run model(rbnn) with random parameter
rand_Cp = net(Rand_Turbin_Param);

%find max Cp & Turbin Param
[max_Cp, Index] = max(rand_Cp);
max_Cp;
Index;

Rand_Turbin_Param(:,Index);

%Paper Opt Paramter
Turbin_Param_Opt = [1.306;1.129;1.109;0.489];
Cp_Opt = net(Turbin_Param_Opt);