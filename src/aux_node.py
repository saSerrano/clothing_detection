import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from clothing_detection.msg import Box, BoxArray
from cv_bridge import CvBridge, CvBridgeError
import cv2

def main():
    rospy.init_node('aux_node', anonymous=True)

    image_pub = rospy.Publisher("/clothing_detector/input_image",Image,queue_size=1)
    bridge = CvBridge()

    # Load image
    cv_image = cv2.imread('/home/sergio/code/ailia-models/deep_fashion/clothing-detection/input.jpg')
    cv2.imshow('window',cv_image)
    cv2.waitKey(0)

    # Publish image
    try:
        tmp = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        image_pub.publish(tmp)
    except CvBridgeError as e:
        print(e)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()