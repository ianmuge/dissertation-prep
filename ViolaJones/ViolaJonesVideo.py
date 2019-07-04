import cv2 as cv

video_capture = cv.VideoCapture(0)
while True:
    ret,base_image = video_capture.read()
    grayscale_image = cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)
    face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
    # eye_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    eye_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(
            grayscale_image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    )
    print("Found {0} Faces!".format(len(faces)))

    idx=0
    for (x, y, w, h) in faces:
        cv.rectangle(base_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.putText(base_image, str(idx), (x+10, y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        roi_gray = grayscale_image[y:y + h, x:x + w]
        roi_color = base_image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)
        idx+=1
    cv.imshow('Video', base_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv.destroyAllWindows()
