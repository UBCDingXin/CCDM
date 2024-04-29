close all;
clear; clc


block_sz = 32;
dataset_name = 'steeringangle';  datadir_base = 'fake_data/'; train_type = 'all'; 

% load model
model_name = ['model_whole_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
model_path = ['models/', model_name];
load(model_path);


q = 0.8;
badfake_dir = [datadir_base, 'badfake_images_niqe_', num2str(q), '/'];
[~, ~, ~] = mkdir(badfake_dir);

img_dir = [datadir_base, 'fake_images_for_NIQE/'];
imgs = dir(img_dir);
imgs = imgs(3:end);

N=length(imgs);
niqe_of_each_img = zeros(N,1);

tic;
parfor i = 1:N
    img_name = imgs(i).name;
    img = imread(fullfile(img_dir, img_name));
    niqe_of_each_img(i) = niqe(img, model);
    % fprintf('i=%d, NIQE=%.3f; \n', i, niqe_of_each_img(i));
end
toc 

niqe_quantile = quantile(niqe_of_each_img, q);
fprintf('%.3f-quantile is %.3f; \n', q, niqe_quantile);

parfor i = 1:N
    img_name = imgs(i).name;
    if niqe_of_each_img(i)>niqe_quantile
        status = copyfile(fullfile(img_dir, img_name), badfake_dir);
    end
end

quit()