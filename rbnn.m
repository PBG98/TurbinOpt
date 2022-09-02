%Input Data
Hydro_Param = readtable('hydroparam.csv');
Hydro_Param = table2array(Hydro_Param);
Turbin_Param = Hydro_Param(:, 1:4);

%Data Division
ref_B_tip = Hydro_Param(:,1);
ref_B_70 = Hydro_Param(:,2);
ref_A_tip = Hydro_Param(:,3);
ref_A_70 = Hydro_Param(:,4);
ref_Cp = Hydro_Param(:, 5);

%Transpose Data
Turbin_Param = transpose(Turbin_Param);
ref_Cp = transpose(ref_Cp);

%rbnn param
eg = 0.0001;
sc = 100;

%build model(rbnn)
model_rbnn = newrb(Turbin_Param, ref_Cp);


%lhs random parameter
lhs_B_tip = lhsdesign_modified(480000, 0.901,1.306);
lhs_B_70 = lhsdesign_modified(480000, 00.913,1.198);
lhs_A_tip = lhsdesign_modified(480000, 0.497,1.553);
lhs_A_70 = lhsdesign_modified(480000, 0.489,1.466);
lhs_turbin_param = horzcat(lhs_B_tip, lhs_B_70, lhs_A_tip, lhs_A_70);
lhs_turbin_param = transpose(lhs_turbin_param);

%run model(rbnn) with random parameter
rand_Cp = model_rbnn(lhs_turbin_param);

%find max Cp & Turbin Param
[max_Cp, Index] = max(rand_Cp);

%max_Cp and max_turbin_param
max_Cp;
max_turbin_param = lhs_turbin_param(:,Index);

%Opt Paramter by reference(paper)
Turbin_Param_Opt = [1.306;1.129;1.109;0.489];
Cp_Opt_ref = model_rbnn(Turbin_Param_Opt);