
python imgs_to_groups_fake.py --imgs_dir .\fake_data\fake_images_for_NIQE --out_dir_base .\fake_data

matlab -noFigureWindows -nodesktop -logfile output.txt -r "run Intra_niqe_test_rc49.m" %*

@REM python badfake_imgs_to_h5.py --imgs_dir .\fake_data\badfake_images_niqe_0.9 --quantile 0.9 --out_dir_base .\fake_data --dataset_name RC_badfake %*