""" install.sh

This script make the librairies updates needed to install 
Caffe. This version of Caffe is slightly different from the 
official release by adding the layer image_pair_data_layer

Basically following :
https://github.com/intel/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide
https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/

Written by Sébastien Ohleyer
"""

# Check Ubuntu version
lsb_release -a # 16.04

# Check opencv version
pkg-config --modversion opencv # 3.2.0



echo "apt-get update and upgrade"
echo "####################################"
sudo apt-get update
sudo apt-get upgrade


echo "Install librairies"
echo "####################################"
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get --yes install libatlas-base-dev


echo "Make dependencies"
echo "####################################"
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;

ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.1.0 /usr/lib/x86_64-linux-gnu/libhdf5.so



echo "Verify Makefile.config"
echo "####################################"
# USE_CPU and OPEN_CV
vi Makefile.config.example
cp Makefile.config.example Makefile.config


# Compile Caffe
echo "Start compiling Caffe"
make all
make test
make runtest
make distribute
make pycaffe
