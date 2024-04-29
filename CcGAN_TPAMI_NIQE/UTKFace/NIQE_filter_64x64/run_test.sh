cd fake_data
unzip fake_images.zip
cd ..

bash ./imgs_to_groups_fake.sh

mkdir -p results

matlab -nodisplay -nodesktop -r "run Intra_niqe_test_utkface.m"

cd fake_data
rm -rf fake_images*
