
matlab -noFigureWindows -nodesktop -logfile output.txt -r "run niqe_test_steeringangle.m" %*


@REM set q=0.9
@REM python badfake_imgs_to_h5.py --imgs_dir .\fake_data\badfake_images_niqe_%q% --quantile %q% --out_dir_base .\fake_data --dataset_name SA_badfake %*

