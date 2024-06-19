
# ROS Clothing Detection 

A package for the detection of clothings in images. This project is based on the [ailia models repository](https://github.com/axinc-ai/ailia-models/tree/master/deep_fashion/clothing-detection).

## Installation

1. Create and activate a [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment:
```
conda create -n clothing_det_env python=3.8
conda activate clothing_det_env
```

2. Install clothing-detector dependencies:
```
pip install ailia
git clone https://github.com/axinc-ai/ailia-models.git
cd ailia-models
pip install -r requirements.txt

```

3. Install clothing-detector dependencies:
```
pip install rospy rospkg pyyaml
```

## Run Detector
```
conda activate clothing_det_env
roscd clothing_detection/src
python python clothing_detector_node.py
```