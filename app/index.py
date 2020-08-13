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
    #file = open('image.png','wb')
    #file.write(image)
    #file.close()

    image = Image.open(io.BytesIO(image))
    #image = cv2.imread('image.png')
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.array(image)
    cv2.imwrite('2.jpg',image)
    boxes, _ = face_detector.detect(image)

    box = boxes[0]
    print(box)
    face = extract_face(image, box).unsqueeze(0).to(device)

    features = feature_extractor(face).detach().cpu().numpy()

    print(data['id'])

    # generate image features
    features = features[0].tolist()
    print(type(features))
    print(len(features))
    sio.emit('employee_cover_processed', {
        'features': features,  # todo: alter that list with features one
        'id': data['id']
    })


@sio.on('process_snap')
def on_message(data):
    image = data['snap']
    persons = data['vectors']

    vectors = []
    ids = []
    names = []
    for person in persons:
        vectors.append(person['features'])
        names.append(person['name'])
        ids.append([person['id']])

    vectors = torch.tensor(vectors).to(device)
    image = base64.b64decode(image)
    # processing the image

    image = np.array(Image.open(io.BytesIO(image)))
    if image.shape[2]>2:
        image = image[:,:,:3]
        
    boxes, _ = face_detector.detect(image)
    faces = []
    for box in boxes:

        face = extract_face(image, box)
        faces.append(face)

    faces = torch.stack(faces).to(device)

    features = feature_extractor(faces).detach()
    matchs = []
    found_ids = []
    for feature in features:
        cosine_sim = F.cosine_similarity(feature.unsqueeze(0),vectors)
        sim_idx = cosine_sim.argmax().item()
        if cosine_sim[sim_idx]>DISTANCE_THRESHOLD:
            matchs.append(sim_idx)
            found_ids.append(ids[sim_idx])
        else:
            matchs.append(-1)

    image = np.array(image,np.uint8)
    for i,box in enumerate(boxes):
        cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2 )
        cv2.putText(image,str(names[matchs[i]]),(int(box[0]),int(box[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)



    encoced_image = base64.b64encode(image)


    print(image.shape)

    # todo: emit the image to the client side via socket.io
    sio.emit('employee_found', {
        'snap': encoced_image,  # base64 encoded image,
        'id': found_ids  # id of found employee
    })


@sio.on('connected_successfully')
def on_message(data):
    print(data)


url = 'http://f3348eed8515.ngrok.io'
sio.connect(url + '?service=ai')
print("listening to socket.io server: ")
