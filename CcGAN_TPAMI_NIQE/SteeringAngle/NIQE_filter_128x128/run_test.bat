
@REM matlab -noFigureWindows -nodesktop -logfile output.txt -r "run niqe_test_steeringangle.m" %*

python badfake_imgs_to_h5.py --imgs_dir .\fake_data\badfake_images_niqe_0.8 --quantile 0.8 --out_dir_base .\fake_data --dataset_name SA_badfake %*

