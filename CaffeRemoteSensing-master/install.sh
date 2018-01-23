# Check Ubuntu version
lsb_release -a # 16.04

# Check opencv version
pkg-config --modversion opencv # 3.2.0



# Install librairies
find . -type f -exec sed -i -e 's^"hdf5/serial/hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5/serial/hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;


# Create dependencies
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so.10.0.2 /usr/lib/x86_64-linux-gnu/libhdf5_hl.so
ln -s /usr/lib/x86_64-linux-gnu/libhdf5_serial.so.10.1.0 /usr/lib/x86_64-linux-gnu/libhdf5.so

sudo apt-get update
sudo apt-get --yes install libatlas-base-dev

# if --env keras
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libleveldb-dev

# Copy the Makefile config
# USE_CPU and OPEN_CV
vi Makefile.config.example
cp Makefile.config.example Makefile.config


# Compile Caffe
echo "Start compiling Caffe"
make all
make test
make runtest
make pycaffe
make distribute
