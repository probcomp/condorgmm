path="assets/TUM_RGBD"
mkdir -p $path
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
tar -xvzf rgbd_dataset_freiburg1_desk.tgz -C $path
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
tar -xvzf rgbd_dataset_freiburg1_desk2.tgz -C $path
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
tar -xvzf rgbd_dataset_freiburg1_room.tgz -C $path
wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz
tar -xvzf rgbd_dataset_freiburg2_xyz.tgz -C $path
wget https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar -xvzf rgbd_dataset_freiburg3_long_office_household.tgz -C $path

rm rgbd_dataset_freiburg1_desk.tgz
rm rgbd_dataset_freiburg1_desk2.tgz
rm rgbd_dataset_freiburg1_room.tgz
rm rgbd_dataset_freiburg2_xyz.tgz
rm rgbd_dataset_freiburg3_long_office_household.tgz
