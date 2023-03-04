# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# https://github.com/murtazahassan/Tello-Object-Tracking

from djitellopy import Tello
import cv2
import numpy as np
import time
from simple_pid import PID

class Drone(object):

    # Speed of the drone
    # S = 60
    # Frames per second of the pygame window display
    # A low number also results in input lag, as input information is processed once per frame.
    FPS = 120
    fbRange = [4000, 8000]
    pid = [0.4, 0.4, 0]
    pError = 0

    area = 0
    center_x = 0
    center_y = 0

    drone_cc = 0
    drone_ud = 0
    drone_fb = 0

    pid_cc = PID(0.35, 0.2, 0.2, setpoint=0, output_limits=(-100, 100))
    pid_ud = PID(0.3, 0.3, 0.3, setpoint=0, output_limits=(-80, 80))
    pid_fb = PID(0.35, 0.2, 0.3, setpoint=0, output_limits=(-50, 50))


    def __init__(self, useDrone = False, remainStatic = False):
        self.useDrone = useDrone
        self.remainStatic = remainStatic  # 0 (false) fly, 1 (true) stay in place

        self.frameWidth, self.frameHeight = 640, 480  # weight and height 
        self.deadZone = 100

        # define the lower and upper boundaries of the "green" ball in the HSV color space, 
        self.greenLower = (24, 85, 6) 
        self.greenUpper = (64, 255, 255)
            # lower_red = np.array([160,50,50])
            # upper_red = np.array([180,255,255])  

        if useDrone:
            self.drone = Tello()
            self.setInitialVelocities()
            # Drone velocities between -100~100
            # self.for_back_velocity = 0
            # self.left_right_velocity = 0
            # self.up_down_velocity = 0
            # self.yaw_velocity = 0
            # self.speed = 10
        else:
            self.webcam = cv2.VideoCapture(0)
            self.webcam.set(3, self.frameWidth)
            self.webcam.set(4, self.frameHeight)
 

    def empty(self):
        pass

    def run(self):
        cv2.namedWindow("HSV")
        cv2.resizeWindow("HSV", self.frameWidth, self.frameHeight // 2)
        cv2.createTrackbar("HUE Min","HSV",20,179,self.empty)
        cv2.createTrackbar("HUE Max","HSV",40,179,self.empty)
        cv2.createTrackbar("SAT Min","HSV",148,255,self.empty)
        cv2.createTrackbar("SAT Max","HSV",255,255,self.empty)
        cv2.createTrackbar("VALUE Min","HSV",89,255,self.empty)
        cv2.createTrackbar("VALUE Max","HSV",255,255,self.empty)

        # then initialize the list of tracked points
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", self.frameWidth, self.frameHeight // 2)
        cv2.createTrackbar("Threshold1","Parameters",166,255, self.empty)
        cv2.createTrackbar("Threshold2","Parameters",171,255, self.empty)

        while True: # keep looping :q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif self.useDrone:
                print(self.drone.get_battery())
                if (self.remainStatic == False) :
                    self.drone.takeoff()
                    self.drone.send_rc_control(0, 0, 10, 0)
                    time.sleep(2)
                    self.remainStatic = True  
                    self.drone.send_rc_control(0, 0, 0, 0)
                self.frame = self.getFrame() # grab and resize frame
                if self.frame is True:
                    break
                text =  "Battery: {}%".format(self.drone.get_battery())
            else:
                 _, self.frame = self.webcam.read()
                 text = "Using Webcam"
            
            self.imgContour = self.frame.copy()
            blurred = cv2.GaussianBlur(self.frame, (11, 11), 0)
            imgHsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # convert to HSV color space
            # cv2.imshow('Gaussian', blurred)

            # Construct mask for the colour green
            mask = cv2.inRange(imgHsv, self.greenLower, self.greenUpper )   
            cv2.imshow('Mask', mask)
            result = cv2.bitwise_and(self.frame, self.frame, mask = mask)
            cv2.imshow('Result', result)
            kernel = np.ones((5, 5))
            erosion =  cv2.erode(mask, kernel,  iterations=2)
            self.imgDilation=cv2.dilate(erosion, kernel, iterations=2)
           
            location = self.getContours(self.imgDilation)
            self.display(self.imgContour)
        
            if self.useDrone:
                self.pError = self.trackObject(location, self.pError)
            # self.pError = self.trackObject(location, self.pError) if self.useDrone else None
            cv2.putText(result, text, (self.frameWidth//2,self.frameHeight//8) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stack = self.stackImages(0.4, ([self.frame, result], [self.imgDilation, self.imgContour]))
            cv2.imshow('Horizontal Stacking', stack)
            # time.sleep(1 / self.FPS)

        print('EXITING...')
        if self.useDrone:
            self.drone.land() 
            self.drone.end()
        else:
            self.webcam.release()
        cv2.destroyAllWindows()

    def trackObject(self, location, pError):
        if location is None:
            self.drone.yaw_velocity = 0
            self.drone.for_back_velocity = 0
            self.drone.up_down_velocity = 0
            return 0
        # if self.center_x is not None and self.center_y is not None:
        center_x, center_y = location[0]
        # center_y = center_y - 20
        area = location[1]
        direction = location[2]

        if center_x != 0:
            error = center_x - self.frameWidth // 2
            speed = self.pid[0] * error + self.pid[1] * (error - self.pError)
            speed = int(np.clip(speed, -100, 100))
            # print('yaw_velocity: ', speed)
            self.drone.yaw_velocity = speed
            

            # if direction[0] is True:
            #     self.yaw_velocity = -35
            # elif direction[1] is Trueq:
            #     self.yaw_velocity = 35

            if self.fbRange[0] < area < self.fbRange[1]:
                print('for_back_velocity: ', area)
                self.drone.for_back_velocity = 0
                self.text = 'DONT'
            elif area > self.fbRange[1]:
                print('for_back_velocity: ', -20)
                self.drone.for_back_velocity = -20
                self.text = 'MOVE BACKWARD'
            elif area < self.fbRange[0] and area != 0:
                print('for_back_velocity: ', 20)
                self.text = 'MOVE FORWARD'
                self.drone.for_back_velocity = 20;
        else:
            self.text = 'NO'
            self.drone.for_back_velocity = 0
            self.drone.yaw_velocity = 0
            error = 0

        if center_y != 0:
            if direction[2] is True:
                print('up down: -20')
                self.drone.up_down_velocity = 25
            elif direction[3] is True:
                print('ip down: 20')
                self.drone.up_down_velocity = -25
            else:
                print('up down: 0')
                self.drone.up_down_velocity = 0
        else:
            print('up down 0')
            self.drone.up_down_velocity = 0

        self.drone.send_rc_control(0, self.drone.for_back_velocity, self.drone.up_down_velocity, self.drone.yaw_velocity)
        return error
        # if self.direction == 1:
        #     self.yaw_velocity = -35
        # elif self.direction == 2:
        #     self.yaw_velocity = 35
        # elif self.direction == 3:
        #     self.up_down_velocity= 35
        # elif self.direction == 4:
        #     self.up_down_velocity= -35
        # else:
        #     self.left_right_velocity = 0; self.for_back_velocity = 0;self.up_down_velocity = 0; self.yaw_velocity = 0
        # # SEND VELOCITY VALUES TO TELLO
        # # if self.drone.send_rc_control:
        # self.drone.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity, self.yaw_velocity)
        # print(self.direction)

    
    def setInitialVelocities(self):
        self.drone.connect()
        self.drone.for_back_velocity = 0
        self.drone.left_right_velocity = 0
        self.drone.up_down_velocity = 0
        self.drone.yaw_velocity = 0
        self.drone.speed = 0
        # self.drone.set_speed(self.speed)
        # In case streaming is on. This happens when we quit this program without the escape key.
        self.drone.streamoff()
        self.drone.streamon()

    def getFrame(self):
        frame_read = self.drone.get_frame_read()
        if frame_read.stopped:
            return frame_read.stoppped
        frame = frame_read.frame
        frame  = cv2.resize(frame, (self.frameWidth, self.frameHeight))
        return frame


    def getContours(self, img):
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
       
        if len(contours) > 0:
            cnt = max(contours, key = cv2.contourArea)
            ((self.center_x, self.center_y), radius) = cv2.minEnclosingCircle(cnt)
            # self.area = cv2.contourArea(cnt)
            direction = [False] * 4
            if radius > 0:
                cv2.drawContours(self.imgContour, cnt, -1, (255, 0, 255), 7)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                x , y , w, h = cv2.boundingRect(approx)
                self.area = w * h
                cx = int(x + (w / 2))   # center x of the object
                cy = int(y + (h / 2))   # center y of the object
                print(x,y,w,h)
                
                if (cx < self.frameWidth//2 - self.deadZone):
                    # cx_min = cx
                    cv2.putText(self.imgContour, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    cv2.rectangle(self.imgContour,(0,int(self.frameHeight/2-self.deadZone)),(int(self.frameWidth/2)-self.deadZone,int(self.frameHeight/2)+self.deadZone),(0,0,255),cv2.FILLED)
                    direction[0] = True
                if (cx > self.frameWidth//2 + self.deadZone):
                    # cx_max = cx
                    cv2.putText(self.imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    cv2.rectangle(self.imgContour,(int(self.frameWidth/2+self.deadZone),int(self.frameHeight/2-self.deadZone)),(self.frameWidth,int(self.frameHeight/2)+self.deadZone),(0,0,255),cv2.FILLED)
                    direction[1] = True
                if (cy < self.frameHeight //2 - self.deadZone):
                    # cy_min = cy
                    cv2.putText(self.imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    cv2.rectangle(self.imgContour,(int(self.frameWidth/2-self.deadZone),0),(int(self.frameWidth/2+self.deadZone),int(self.frameHeight/2)-self.deadZone),(0,0,255),cv2.FILLED)
                    # self.direction = 3
                    direction[2] = True
                if (cy > self.frameHeight //2  + self.deadZone):
                    # cy_max = cy
                    cv2.putText(self.imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
                    cv2.rectangle(self.imgContour,(int(self.frameWidth/2-self.deadZone),int(self.frameHeight/2)+self.deadZone),(int(self.frameWidth/2+self.deadZone),self.frameHeight),(0,0,255),cv2.FILLED)
                    # self.direction = 4
                    direction[3] = True
                # print(direction)
                center = (int(self.center_x),int(self.center_y))

                cv2.circle(self.imgContour, (int(self.center_x), int(self.center_y)), int(radius),(0, 255, 255), 6)
                cv2.circle(self.imgContour, center, 5, (0, 0, 255), -1) 
                cv2.line(self.imgContour, (int(self.frameWidth/2),int(self.frameHeight/2)), (cx,cy),(0, 0, 255), 3)
                cv2.rectangle(self.imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(self.imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,(0, 255, 0), 2)
                cv2.putText(self.imgContour, "Area: " + str(int(self.area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
                cv2.putText(self.imgContour, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,0.7,(0, 255, 0), 2)
                print(self.area)
                print("\n")
                return ([self.center_x, self.center_y], self.area, direction)
            else:
                return ([0,0], 0, direction)
        return None

    def display(self, img):
        cv2.circle(img,((self.frameWidth//2),(self.frameHeight//2)),5,(0,0,255),5)
        cv2.line(img,((self.frameWidth//2)-self.deadZone,0),((self.frameWidth//2)-self.deadZone,self.frameHeight),(255,255,0),3)
        cv2.line(img,((self.frameWidth//2)+self.deadZone,0),((self.frameWidth//2)+self.deadZone,self.frameHeight),(255,255,0),3)
        cv2.line(img, (0,(self.frameHeight // 2) - self.deadZone), (self.frameWidth,(self.frameHeight // 2) - self.deadZone), (255, 255, 0), 3)
        cv2.line(img, (0,(self.frameHeight // 2) + self.deadZone), (self.frameWidth, (self.frameHeight // 2) + self.deadZone), (255, 255, 0), 3)

    def stackImages(self, scale,imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor                                                                                                               
        return ver

if __name__ == "__main__":
    drone = Drone(useDrone= False, remainStatic= True)
    drone.run()

