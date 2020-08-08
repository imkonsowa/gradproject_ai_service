import socketio
import base64

# todo: initialize the AI model
model = ''

sio = socketio.Client()


@sio.on('process_employee_cover')
def on_message(data):
    image = base64.b64decode(data['snap'])
    print(data['id'])

    # todo: generate image features

    sio.emit('employee_cover_processed', {
        'features': [x for x in range(128)],  # todo: alter that list with features one
        'id': data['id']
    })


@sio.on('process_snap')
def on_message(data):
    image = data['snap']
    vectors = data['vectors']

    # the image as a file
    image = base64.b64decode(image)

    # todo: process the image

    # todo: emit the image to the client side via socket.io
    sio.emit('employee_found', {
        'snap': '',  # base64 encoded image,
        'id': ''  # id of found employee
    })


@sio.on('connected_successfully')
def on_message(data):
    print(data)


url = 'http://f3d2cb083152.ngrok.io'
sio.connect(url + '?service=ai')
print("listening to socket.io server: ")
