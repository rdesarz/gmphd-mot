# Multiple object tracking for ROS

This project provides a Python and a C++ (not yet implemented) library for object tracking applications. It will also contain examples to show how it can be integrated into a ROS project.  

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project is based on ROS and is currently developped for ROS melodic. Check http://wiki.ros.org/melodic/Installation for the installation instruction 

### Installation

First step is to clone this repository into your ROS workspace directory:

```
git clone https://github.com/rdesarz/motlib.git
```

Then you have to build motlib so that the ROS project can link with it. Please make sure to create a build folder in the root of the project as it is required
when building the ROS part.

```
mkdir build
cd build
cmake ../
make -j4
```

Then move to the `ros` folder of the repo and run `catkin_make` to build the package 

```
cd ros
catkin_make  
```

## Example

### 2D multiple non-extended target 

An ROS launch file provides a tracking example of multiple single points target in a squared field of view. You can run it using the following command:

```
roslaunch motlib_ros target_tracking_example.launch
```

## Currently working on
This project is a work in progress. The first step is to provide a fast and efficient C++ implementation of the first algorithm (Gaussian Mixture PHD filter). 

## Authors

* **Romain Desarzens** - *Initial work* - [rdesarz](https://github.com/rdesarz)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
