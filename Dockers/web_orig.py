import os
import shutil
from flask import Flask, render_template, request, \
    Response, redirect, url_for
from camera import Camera
from support import util

app = Flask(__name__)
camera = None
mail_server = None
mail_conf = "static/mail_conf.json"

def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


@app.route('/')
def root():
    return redirect(url_for('index'))

@app.route('/index/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_feed()
        
        #frame = util.findandret(frame) 
        
        #yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture/')
def capture():
    camera = get_camera()
    stamp = camera.capture()
    return redirect(url_for('show_capture', timestamp=stamp))

def stamp_file(timestamp):
    return 'captures/' + timestamp +".jpg"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
