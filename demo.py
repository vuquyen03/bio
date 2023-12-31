import tensorflow as tf 
import numpy as np
import cv2
import detect_face
import time

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709

def to_rgb (img):
    w, h = img.shape
    ret = np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,0] = ret[:,:,1]=ret[:,:,2]=img
    return ret

print ("Creating networks and loading parameters")
gpu_memory_fraction = 1.0
with tf.Graph().as_default():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs found!")
    
    sess = tf.compat.v1.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame by frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if gray.ndim == 2:
        img = to_rgb(gray)
        
    start = time.time()
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    end = time.time()
    print('Current frame processing time: %.2fms'%((end-start))*1000)
    nrof_faces = bounding_boxes.shape[0]
    
    for face_postion in bounding_boxes:
        face_postion = face_postion.astype(int)
        
        cv2.rectangle(frame,
                      (face_postion[0], face_postion[1]),
                      (face_postion[2], face_postion[3]),
                      (0,255,0), 2)
        
    cv2.imshow('MTCNN Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()