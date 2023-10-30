clc;
clear;
data = load("data_1.mat");
loc_label = data.train_location_label;
act_label = data.train_activity_label;
csi_data = data.train_data;
data = load("data_2.mat");
loc_label = [loc_label;data.train_location_label];
act_label = [act_label;data.train_activity_label];
csi_data = [csi_data;data.train_data;];


num = 960;
cv = 5;
divide = []
for j=1:cv
for i=1:num/cv
    divide = [divide,j];
end
end

index = randperm(num);
divide = divide(index);

for j = 1:cv
train_location_label = [];
train_activity_label = [];
train_data = [];

test_location_label = [];
test_activity_label = [];
test_data = [];

for i=1:num
    if divide(i)~=j
    train_location_label = [train_location_label;loc_label(i)];
    train_activity_label = [train_activity_label;act_label(i)];
    train_data = [train_data;csi_data(i,:,:)];
    else
    test_location_label = [test_location_label;loc_label(i)];
    test_activity_label = [test_activity_label;act_label(i)];
    test_data = [test_data;csi_data(i,:,:)];
    end
end

% train_data_split_amp = struct('train_location_label',{train_location_label},'train_activity_label',{train_activity_label},'train_data',{train_data});
% test_data_split_amp = struct('test_location_label',{test_location_label},'test_activity_label',{test_activity_label},'test_data',{test_data});

save(['./cvdataset/',num2str(j),'/train_data_split_amp.mat'],'train_location_label','train_activity_label','train_data');
save(['./cvdataset/',num2str(j),'/test_data_split_amp.mat'],'test_location_label','test_activity_label','test_data');

end