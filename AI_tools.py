from flask import *
from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)





def gen_frames():  # generate frame by frame from camera
    
    #load the model
    model=tf.keras.models.load_model('Mask_detector_model.h5')

    #loading the cascades
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')



    camera = cv2.VideoCapture(0)  # use 0 for web camera
    #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        
        faces=face_cascade.detectMultiScale(frame,1.3,5)
        
        
        if not success:
            break
        else:
        		   
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                face=frame[y:y+h,x:x+w]
                cropped_face=face
    
                if type(face) is np.ndarray:
                    face=cv2.resize(face,(224,224))
                    im=Image.fromarray(face,'RGB')
                    img_array=np.array(im)
                    img_array=np.expand_dims(img_array,axis=0)
                    pred=model.predict(img_array)
                    print(pred)
            
                    if(pred[0][0]>0.5):
                        prediction='Mask'
                        cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    else:
                        prediction='No Mask'
                        cv2.putText(cropped_face,prediction,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                else:
                    cv2.putText(frame,'No Face Found',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            			

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break	
    video_capture.release()
    cv2.destroyAllWindows()	
	
	    
@app.route('/video')
def video():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/streaming')
def home():
    """Video streaming home page."""
    return render_template('streaming.html')





def predictCOVID(imageToBeTested):

    model = load_model("covid_initial.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    image = cv2.imread(imageToBeTested)
    image = cv2.resize(image, (300, 300))
    image = np.reshape(image, [1, 300, 300, 3])

    classes = model.predict_classes(image)  # [[0]]

    label = ["COVID-19 INFECTED", "NORMAL"]

    return label[classes[0][0]]


@app.route('/predict')
def index():
    return render_template("predict.html")

@app.route('/upload-image', methods=['POST'])
def uploadImage():
    if request.method == 'POST': # Just to Validate if user is uploading the file in POST Request
        file = request.files['image']
        file.save(file.filename)

        label = predictCOVID(file.filename)

        return render_template('result.html', name=label)



@app.route('/')
def new():
    return render_template("index.html")




if __name__ == '__main__':
    # app.run() # execute the app i.e. let the app run on Flask Server
    app.run(host='localhost' , port='5000',debug='true')     # Enable Debugging for the error

