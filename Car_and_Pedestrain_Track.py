import cv2

# the image
img_file = 'car_road.jpeg'
video = cv2.VideoCapture('vid.mp4')
#video = cv2.VideoCapture(0)


# pre-trained car and pedestrian classifier
car_tracker_file = 'cars_detector.xml'
pedestrian_tracker_file = ('haarcascade_fullbody.xml')


# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)




#run until car stops / any other actions happens
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    #successful check 
    if read_successful:
        #first convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrain
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangle around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


     # draw rectangle around pedestrians
    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    

     
    # Display the image with the faces spotted
    cv2.imshow('Car and Pedestrain Detector', frame)


    # Wait and Listens for a key press to close
    key = cv2.waitKey(1)
    
    #Stop if Q key is pressed
    if key==81 or key==113:
        break
#Release the VideoCapture object
video.release()


print("Code Completed")


























""""
import cv2

# the image
img_file = 'car_road.jpeg'

# pre-trained car classifier
classifier_file = 'cars_detector.xml'

# create opencv image
img = cv2.imread(img_file)

# convert to gray scale(need for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)


# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# draw rectangle around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#car_1 = cars[0]
#(x, y, w, h) = car_1
#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


# Display the image with the faces spotted
cv2.imshow('Car and Pedestrain Detector', img)

# Wait and Listens for a key press to close
cv2.waitKey()



print("Code Completed")

"""