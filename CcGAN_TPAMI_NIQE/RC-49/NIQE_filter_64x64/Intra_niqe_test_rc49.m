% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'rc49';  datadir_base = 'fake_data/fake_images_by_angles/'; train_type = 'all'; %('all', '10')


angles = 0.1: 0.2: 89.9;
N = length(angles);
intra_niqe = zeros(N,1);


q = 0.9;
niqe_quantiles = zeros(N,1);
badfake_dir = ['fake_data/badfake_images_niqe_', num2str(q), '/'];
[~, ~, ~] = mkdir(badfake_dir);


tic;

for i = 1: N

    angle = angles(i);

    badfake_dir_i = [badfake_dir, num2str(angle)];
    [~, ~, ~] = mkdir(badfake_dir_i);

    model_name = ['model_angle_', num2str(angle,'%.1f'), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];
    load(model_path);

    img_dir = [datadir_base, num2str(angle,'%.1f'), '/'];
    imgs = dir(img_dir);
    imgs = imgs(3:end);

    niqe_of_each_img = zeros(length(imgs),1);
    parfor img_idx = 1: length(imgs)
        img_name = imgs(img_idx).name;
        img = imread(fullfile(img_dir, img_name));
        niqe_of_each_img(img_idx) = niqe(img, model); %compute NIQE by pre-trained model
    end
    intra_niqe(i) = mean(niqe_of_each_img);
    niqe_quantiles(i) = quantile(niqe_of_each_img, q);
    toc

    parfor img_idx = 1: length(imgs)
        img_name = imgs(img_idx).name;
        if niqe_of_each_img(img_idx)>niqe_quantiles(i)
            status = copyfile(fullfile(img_dir, img_name), badfake_dir_i);
        end
    end

    fprintf('angle=%.1f, nfake=%d, NIQE=%.3f, %.2f Quantile=%.3f ; \n', angle, length(imgs), intra_niqe(i), q, niqe_quantiles(i));
end
toc

quit()
