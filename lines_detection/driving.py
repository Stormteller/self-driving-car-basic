import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from lines_detection.image_processing import get_road_lines_from_image


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 20
MIN_SPEED = 10

speed_limit = MAX_SPEED

prev_left = 45
prev_right = 45
left_set_counter = 0
right_set_counter = 0

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        print(steering_angle)
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            open_cv_image = np.array(image)[:, :, ::-1].copy()  # ot openCV format, BGR to RGB
            road = get_road_lines_from_image(open_cv_image)
            road_angles = road.get('angels')
            road_lines = road.get('lines')
            print(road_angles)
            steering_angle = 0
            # if road.get('horizontal') and road_angles.get('left') and road_angles.get('right'):
            #     steering_angle = (road_angles.get('left') + road_angles.get('right')) / 100
            # elif not road.get('horizontal') and road_angles.get('left') and road_angles.get('right'):
            #     if np.abs(road_angles.get('left')) > -10 or np.abs(road_angles.get('right')) < 10:
            #         steering_angle = (road_angles.get('left') + road_angles.get('right')) / 200
            #     else:
            #         steering_angle = 0
            # elif not road.get('horizontal') and not road_angles.get('left') and road_angles.get('right'):
            #     if road_angles.get('right') < 15:
            #         steering_angle = -0.1
            #     elif road_angles.get('right') > 70:
            #         steering_angle = -0.01
            #     else:
            #         steering_angle = 0
            # elif not road.get('horizontal') and road_angles.get('left') and not road_angles.get('right'):
            #     if road_angles.get('left') > -15:
            #         steering_angle = 0.1
            #     elif road_angles.get('left') < -70:
            #         steering_angle = 0.01
            #     else:
            #         steering_angle = 0

            # if road['lines']['horizontal'] is not None:
            #     steering_angle = -(road_angles['left'] + road_angles['right']) / 200
            # else:
            global prev_left
            if road_lines['left'] is None:
                road_angles['left'] = prev_left
            else:
                prev_left = road_angles['left']

            global prev_right
            if road_angles['right'] is None:
                road_angles['right'] = prev_right
            else:
                prev_right = road_angles['right']

            print(road_angles)

            if road_lines['left'] is not None and \
                    road_lines['right'] is not None and \
                    np.abs(road_angles['left'] + road_angles['right']) > 0:
                steering_angle = (road_angles['left'] + road_angles['right']) / 200
            elif road_lines['left'] is not None:
                steering_angle = -(road_angles['left'] + 35) / 60 if road_angles['left'] < -35 else 0
            elif road_lines['right'] is not None:
                steering_angle = -(road_angles['right'] - 20) / 80 if road_angles['right'] > 20 else 0





            if steering_angle > 5:
                throttle = 0.1
            else:
                throttle = 1

            if speed >= 10:
                throttle = 0

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            raise e

        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
