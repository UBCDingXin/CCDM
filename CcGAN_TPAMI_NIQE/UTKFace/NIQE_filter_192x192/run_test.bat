@REM cd fake_data
@REM unzip fake_images.zip
@REM cd ..

python imgs_to_groups_fake.py --imgs_dir .\fake_data\fake_images_for_NIQE --out_dir_base .\fake_data ^ %*

mkdir results

matlab -noFigureWindows -nodesktop -logfile output.txt -r "run Intra_niqe_test_utkface.m" %*

@REM python badfake_imgs_to_h5.py --imgs_dir .\fake_data\badfake_images_niqe_0.9 --filter_rho 0.9 --out_dir_base .\fake_data --dataset_name UK_badfake %*