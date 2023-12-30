import os
import time
import uuid
import cv2
from mtcnn import MTCNN

new_height = 256
new_width = 256

def access_camera(camera_number):
    cap = cv2.VideoCapture(camera_number)  

    while True:
        ret, frame = cap.read()

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# # Create folder 
# os.makedirs(path, exist_ok=True)
# number_images = 70

def capture_face(camera_number, number_images, path):
    cap = cv2.VideoCapture(camera_number)
    for imgnum in range(number_images):
        print("Collecting image {}".format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(path, f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def detectfaceMTCNN(input, output, newHeight, newWidth):
    os.makedirs(output, exist_ok=True)
    detector = MTCNN()
    
    for filename in os.listdir(input):
        # Đường dẫn đầy đủ đến tập tin ảnh
        image_path = os.path.join(input, filename)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt
        faces = detector.detect_faces(image_rgb)

        for i, face in enumerate(faces):
            # Lấy thông tin về khung hình chứa khuôn mặt
            x, y, width, height = face['box']

            # Cắt phần mặt từ ảnh gốc
            face_image = image[y:y+height, x:x+width]
            resized_face = cv2.resize(face_image, (newWidth, newHeight))

            # Tạo đường dẫn đến thư mục đích và tên tập tin
            output_filename = f'{filename.split(".")[0]}_face{i}.jpg'
            output_path = os.path.join(output, output_filename)

            cv2.imwrite(output_path, resized_face)
            
            # lấy một ảnh duy nhất (người trung tâm bức ảnh), quần chúng bỏ qua
            if i == 0:
                break
    
# access_camera(0)