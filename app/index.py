import socketio
import base64
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
import numpy as np
from PIL import Image
import io
import torch.nn.functional as F

import cv2
# initializing the AI models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_detector = MTCNN(keep_all=True,device = device).eval().to(device)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval().to(device)
DISTANCE_THRESHOLD = .5

sio = socketio.Client()


@sio.on('process_employee_cover')
def on_message(data):
    image = base64.b64decode(data['snap'])
    image = np.array(Image.open(io.BytesIO(image)))
    boxes, _ = face_detector.detect(image)
    box = boxes[0]
    face = extract_face(image, box).to(device)
    features = feature_extractor(face).detach().numpy()

    print(data['id'])

    # generate image features

    sio.emit('employee_cover_processed', {
        'features': [x for x in features[0]],  # todo: alter that list with features one
        'id': data['id']
    })


@sio.on('process_snap')
def on_message(data):
    image = data['snap']
    vectors = data['vectors']
    vectors = torch.tensor(vectors).to(device)
    image = base64.b64decode(image)
    # processing the image

    image = np.array(Image.open(io.BytesIO(image)))
    boxes, _ = face_detector.detect(image)
    faces = []
    for box in boxes:

        face = extract_face(image, box)
        faces.append(face)

    faces = torch.stack(faces).to(device)

    features = feature_extractor(faces).detach()
    matchs = []
    for feature in features:
        cosine_sim = F.cosine_similarity(feature.unsqueeze(0),vectors)
        sim_idx = cosine_sim.argmax().item()
        if cosine_sim[sim_idx]>DISTANCE_THRESHOLD:
            matchs.append(sim_idx)
        else:
            matchs.append(-1)

    for i,box in enumerate(boxes):
        cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2 )
        cv2.putText(image,str(matchs[i]),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)



    encoced_image = base64.a85encode(image)



    # todo: emit the image to the client side via socket.io
    sio.emit('employee_found', {
        'snap': encoced_image,  # base64 encoded image,
        'id': ''  # id of found employee
    })


@sio.on('connected_successfully')
def on_message(data):
    print(data)


url = 'http://f3d2cb083152.ngrok.io'
sio.connect(url + '?service=ai')
print("listening to socket.io server: ")
