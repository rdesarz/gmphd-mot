# GM-PHD filter based multiple object tracking

This aim of this project is to develop a library for multiple object tracking. This a work in progress and 
it currently integrates an implementation of the Gaussian Mixture Probability Hypothesis Density filter used to estimate 
states of the target and some dynamic and measurement models. 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installation

This library itself depends on Eigen 3 and was tested only on Ubuntu 18.04 and 20.04. You can download Eigen using `apt`:

```
apt-get install libeigen3-dev
```

The library is stored in folder `motlib_cpp`as a standalone CMake project. To install it first clone the project and run 
`make`

```
git clone https://github.com/rdesarz/motlib.git
cd motlib/motlib_cpp
mkdir build 
cd build
cmake -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ..
sudo make install
```

A ROS package is also provided with examples. It currently provides a simple 2D state estimation example with random 
single point targets. You will need ROS Noetic to run it. Please check http://wiki.ros.org/noetic/Installation for the 
installation instruction.

Then from the root folder, move to `motlib_ros` folder, run `catkin_make` to build the package 

```
cd motlib_ros
catkin_make  
```

## Example

### 2D multiple non-extended target 

An ROS launch file provides a tracking example of multiple single points target in a squared field of view. You can run it using the following command in `motlib_ros` folder:

```
source devel/setup.bash
roslaunch motlib_ros target_tracking_example.launch
```

## Authors

* **Romain Desarzens** - *Initial work* - [rdesarz](https://github.com/rdesarz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
