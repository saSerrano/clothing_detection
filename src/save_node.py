import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from clothing_detection.msg import Box, BoxArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys

class image_converter:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/clothing_detector/results",BoxArray,self.callback)
        self.cv_image = cv2.imread('/home/sergio/code/ailia-models/deep_fashion/clothing-detection/input.jpg')

    def callback(self,data):
        img = self.cv_image.copy()
        # Draw boxes
        for b in data.boxes:
            print(b.class_id)
            print(b.prob)
            print('-----------')
            img = cv2.rectangle(img,(b.x,b.y),(b.x+b.width,b.y+b.height),(255,0,0),2)
        
        cv2.imshow('save window',img)
        cv2.waitKey(0)

        # Save the image
        cv2.imwrite('/home/sergio/catkin_ws/src/clothing_detection/src/output.png',img)
        print('Saved image at /home/sergio/catkin_ws/src/clothing_detection/src/output.png')


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)