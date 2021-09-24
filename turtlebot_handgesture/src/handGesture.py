#!/usr/bin/python3
import cv2
import mediapipe as mp
import time
import numpy as np
import math as m 



import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.002
ANG_VEL_STEP_SIZE = 0.02


cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                 output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''
    
    # Get the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image to write the count of fingers on.
    output_image = image.copy()
    
    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}
    
    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mpHands.HandLandmark.INDEX_FINGER_TIP, mpHands.HandLandmark.MIDDLE_FINGER_TIP,
                        mpHands.HandLandmark.RING_FINGER_TIP, mpHands.HandLandmark.PINKY_TIP]
    
    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    
    
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):
        
        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        
        # Retrieve the landmarks of the found hand.
        hand_landmarks =  results.multi_hand_landmarks[hand_index]
        
        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:
            
            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]
            
            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                
                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                
                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1
        
        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP - 2].x
        
        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
            
            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            
            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1
     
    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    if draw:

        # Write the total count of the fingers of both hands on the output image.
        cv2.putText(output_image, " Total Fingers: ", (10, 25),cv2.FONT_HERSHEY_COMPLEX, 1, (20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20,255,155), 10, 10)

    # # Check if the output image is specified to be displayed.
    # if display:
        
    #     # Display the output image.
    #     plt.figure(figsize=[10,10])
    #     plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # # Otherwise
    # else:

    # Return the output image, the status of each finger and the count of the fingers up of both hands.
    return output_image, fingers_statuses, count


def recognizeGestures(image, fingers_statuses, count, draw=True, display=True):
    '''
    This function will determine the gesture of the left and right hand in the image.
    Args:
        image:            The image of the hands on which the hand gesture recognition is required to be performed.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands. 
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        draw:             A boolean value that is if set to true the function writes the gestures of the hands on the
                          output image, after recognition.
        display:          A boolean value that is if set to true the function displays the resultant image and 
                          returns nothing.
    Returns:
        output_image:   A copy of the input image with the left and right hand recognized gestures written if it was 
                        specified.
        hands_gestures: A dictionary containing the recognized gestures of the right and left hand.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Store the labels of both hands in a list.
    hands_labels = ['RIGHT', 'LEFT']
    
    # Initialize a dictionary to store the gestures of both hands in the image.
    hands_gestures = {'RIGHT': "UNKNOWN", 'LEFT': "UNKNOWN"}
    
    # Iterate over the left and right hand.
    for hand_index, hand_label in enumerate(hands_labels):
        
        # Initialize a variable to store the color we will use to write the hands gestures on the image.
        # Initially it is red which represents that the gesture is not recognized.
        color = (0, 0, 255)
        
        # Check if the person is making the 'V' gesture with the hand.
        ####################################################################################################################
        # Check if the number of fingers up is 1 and the fingers that are up are the index fingers
        if count[hand_label] == 1 and fingers_statuses[hand_label+'_INDEX']:
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "BACK"
            
            # Update the color value to green.
            color=(0,255,0)

        # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        if count[hand_label] == 2  and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_INDEX']:
            pass
            # # Update the gesture value of the hand that we are iterating upon to V SIGN.
            # hands_gestures[hand_label] = "V SIGN"
            
            # # Update the color value to green.
            # color=(0,255,0)
            
        ####################################################################################################################            
        
        # Check if the person is making the 'SPIDERMAN' gesture with the hand.
        ##########################################################################################################################################################
        
        # Check if the number of fingers up is 3 and the fingers that are up, are the thumb, index and the pinky finger.
        elif count[hand_label] == 3 and fingers_statuses[hand_label+'_THUMB'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY']:
            pass    
            # # Update the gesture value of the hand that we are iterating upon to SPIDERMAN SIGN.
            # hands_gestures[hand_label] = "SPIDERMAN SIGN"
 
            # # Update the color value to green.
            # color=(0,255,0)
                
        ##########################################################################################################################################################
        
        # Check if the person is making the 'HIGH-FIVE' gesture with the hand.
        ####################################################################################################################
        
        # Check if the number of fingers up is 5, which means that all the fingers are up.
        elif count[hand_label] == 5:
            
            # Update the gesture value of the hand that we are iterating upon to HIGH-FIVE SIGN.
            hands_gestures[hand_label] = "OPEN"
            
            # Update the color value to green.
            color=(0,255,0)
       
               # Check if the number of fingers up is 2 and the fingers that are up, are the index and the middle finger.
        elif count[hand_label] == 0:
            
            # Update the gesture value of the hand that we are iterating upon to V SIGN.
            hands_gestures[hand_label] = "CLOSE"
            
            # Update the color value to green.
            color=(0,255,0)
        ####################################################################################################################  
        
        # Check if the hands gestures are specified to be written.
        if draw:
        
            # Write the hand gesture on the output image. 
            cv2.putText(output_image, hand_label +': '+ hands_gestures[hand_label] , (10, (hand_index+1) * 60),
                        cv2.FONT_HERSHEY_PLAIN, 4, color, 5)
            
    
    # # Check if the output image is specified to be displayed.
    # if display:
 
    #     # Display the output image.
    #     plt.figure(figsize=[10,10])
    #     plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # # Otherwise
    # else:
 
    # Return the output image and the gestures of the both hands.
    return output_image, hands_gestures

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(target_linear_vel, target_angular_vel):
    return "currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel)

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min( input, output + slop )
    elif input < output:
        output = max( input, output - slop )
    else:
        output = input

    return output


def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

def checkLinearLimitVelocity(vel):
    if turtlebot3_model == "burger":
      vel = constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)
    elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
      vel = constrain(vel, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)
    else:
      vel = constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)

    return vel

def checkAngularLimitVelocity(vel):
    if turtlebot3_model == "burger":
      vel = constrain(vel, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)
    elif turtlebot3_model == "waffle" or turtlebot3_model == "waffle_pi":
      vel = constrain(vel, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)
    else:
      vel = constrain(vel, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)

    return vel

#actual run
if __name__=="__main__":
    
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('turtlebot_handgesture')
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    turtlebot3_model = rospy.get_param("model", "burger")

    status = 0
    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    control_linear_vel  = 0.0
    control_angular_vel = 0.0



    while(1):
        success, img = cap.read()
        # img = cv2.imread("data/test.jpg", cv2.IMREAD_COLOR)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #flip horizontal
        imgRGB = cv2.flip(imgRGB,1)
        results = hands.process(imgRGB)
        hand_status = None
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(imgRGB, handLms, mpHands.HAND_CONNECTIONS)
            # imgRGB, fingers_statuses, count = countFingers(imgRGB, results, display=False)
            imgRGB, fingers_statuses, count = countFingers(imgRGB, results, draw=False, display = False)    
            imgRGB, hand_status = recognizeGestures(imgRGB, fingers_statuses, count)   


        image = cv2.resize(imgRGB,(1280,780), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", image)
        cv2.waitKey(1)


        #update message!!
        key = getKey()
        if  hand_status == None:
            target_linear_vel   = 0.0
            control_linear_vel  = 0.0
            target_angular_vel  = 0.0
            control_angular_vel = 0.0
            print(vels(target_linear_vel, target_angular_vel)) 
            if (key == '\x03'):
                break

        
        elif hand_status["RIGHT"] == "CLOSE" and hand_status["LEFT"] == "CLOSE":
            target_linear_vel = checkLinearLimitVelocity(target_linear_vel + LIN_VEL_STEP_SIZE)
            status = status + 1
            print(vels(target_linear_vel,target_angular_vel))

        elif hand_status["RIGHT"] == "BACK" and hand_status["LEFT"] == "BACK" :
            target_linear_vel = checkLinearLimitVelocity(target_linear_vel - LIN_VEL_STEP_SIZE)
            status = status + 1
            print(vels(target_linear_vel,target_angular_vel))

        elif hand_status["RIGHT"] == "OPEN" and hand_status["LEFT"] == "CLOSE" :
            target_angular_vel = checkAngularLimitVelocity(target_angular_vel + ANG_VEL_STEP_SIZE)
            status = status + 1
            print(vels(target_linear_vel,target_angular_vel))

        elif hand_status["RIGHT"] == "CLOSE" and hand_status["LEFT"] == "OPEN" :
            target_angular_vel = checkAngularLimitVelocity(target_angular_vel - ANG_VEL_STEP_SIZE)
            status = status + 1
            print(vels(target_linear_vel,target_angular_vel))

        elif key == ' ' or key == 's' :
            target_linear_vel   = 0.0
            control_linear_vel  = 0.0
            target_angular_vel  = 0.0
            control_angular_vel = 0.0
            print(vels(target_linear_vel, target_angular_vel))
        
        elif  hand_status["RIGHT"] == "OPEN" and hand_status["LEFT"] == "OPEN" :
            if (key == '\x03'):
                break

        twist = Twist()

        control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, (LIN_VEL_STEP_SIZE/2.0))
        twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

        control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ANG_VEL_STEP_SIZE/2.0))
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

        pub.publish(twist)


    # except:
    #     print("error")

    # finally:
    #     twist = Twist()
    #     twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
    #     twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
    #     pub.publish(twist)

    # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)