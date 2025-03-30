# Setup guide for AprilTag

[apriltag](https://github.com/AprilRobotics/apriltag) library is officially supported on linux. For this reason an Ubuntu 24.04 environment is used for this setup guide. This guide will help you set up the AprilTag library to be used with Python.

# Install dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install cmake g++ pkg-config python3-dev python3-numpy libeigen3-dev libopencv-dev ninja-build
```

# Clone the apriltag repository

```bash
git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
```

# Create a build directory and compile

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
```

# Add to python path

```bash
echo 'export PYTHONPATH=/usr/local/lib/python3.12/site-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

# Test the installation

```bash
python -c "import apriltag; print(apriltag.__file__)"
```
