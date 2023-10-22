clc;
data = load("data_1.mat");
loc_label = data.train_location_label;
act_label = data.train_activity_label;
csi_data = data.train_data;
num = 600;

train_location_label = [];
train_activity_label = [];
train_data = [];

test_location_label = [];
test_activity_label = [];
test_data = [];


for i=1:num
    if rand(1)>0.2
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

save('train_data_split_amp.mat','train_location_label','train_activity_label','train_data');
save('test_data_split_amp.mat','test_location_label','test_activity_label','test_data);