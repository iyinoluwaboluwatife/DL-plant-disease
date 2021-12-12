#c:\PACKAGES\ds_env\Scripts\python

#%%
from __future__ import division, print_function
import os
import numpy as np
from tensorflow.python.keras.backend import update
from helperFunctions import *
from datetime import datetime

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash, session, Response
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


def new_upload(plant, path, predict, confidence):
    upload = Upload(plant=plant, path=path, predict=predict, confidence=confidence)
    db.session.add(upload)
    db.session.commit()

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plant =  db.Column(db.String(15), nullable=False)
    path =  db.Column(db.String(150), nullable=False)
    date_upload = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    predict = db.Column(db.String(25), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    verified = db.Column(db.Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return f"Post('{self.plant}','{self.predict}',  '{self.confidence}')"


print('Model loaded. Check http://127.0.0.1:5000/')

global capture,rec_frame, grey, switch, neg, face, rec, out, capture_name, plant
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
cwd = os.getcwd()


PLANT_CATEGORIES = {
    'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
                'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 
                'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy'],
    'potato': ['Early Blight', 'Late Blight', 'Healthy'],
    'maize': ['Cercospora Leaf Spot', 'Common Rust', 'Northern Leaf Blight', 'Healthy'],
    'pepper': ['Bacterial Spot', 'Healthy'],
    'cassava': ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy'],
}


PATH = r'C:\Users\IYINOLUWA\Desktop\Deep-Learning-plant_disease_identification'

def gen_frames():  # generate frame by frame from camera
    try:
        # print("\n\n\n \tby gen_frame\n\n")
        global out, capture,rec_frame, capture_name
        while True:
            success, frame = camera.read() 
            if success:
                if(capture):
                    print('here')
                    capture=0
                    stamp= str(datetime.now()).replace('-', '').replace(':', '.')
                    capture_name = f"{plant}_{stamp}.png"
                    # print(f'\n\n{capture_name}\n\n')
                    basepath = os.path.dirname(__file__)
                    default_path = os.path.join(basepath, 'static', 'images', 'image.png')
                    file_path = os.path.join(basepath, 'static', 'images', capture_name)
                    cv2.imwrite(file_path, frame)
                    cv2.imwrite(default_path, frame)
                    camera.release()
                    cv2.destroyAllWindows()
                
                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass
                    
            else:
                pass
    except Exception as e:
        print(e)


global check, what
check = 0
what = 0    

@app.route('/', methods=['GET'])
@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html', page='home')


@app.route('/test', methods=['GET'])
def dropdown():
    plant = ['tomato', 'potato', 'cassava', 'maize']
    tests = ['d', 'potato', 'cassava', 'maize']
    return render_template('test.html', plants=plant, categories=PLANT_CATEGORIES, tests=tests)

@app.route('/all_upload', methods=['GET', 'POST'])
def all_upload():
    page = request.args.get('page', 1, type=int )
    # uploads = Upload.query.order_by(desc(Upload.date_upload)).paginate(page=page, per_page=5)
    uploads = Upload.query.order_by(desc(Upload.date_upload)).all()
    all_uploads = len(uploads)
    # all_uploads = len(uploads)
    if request.method == 'POST':
        try:
            query = request.form['query']
            queries = []
            for what in uploads:
                if query.lower().strip()[:-3] in what.path:
                    print(what.path)
                    queries.append(what)

            all_uploads = len(queries)

            return render_template('all_upload.html', total_count=all_uploads, uploads=queries, categories=PLANT_CATEGORIES)
        except:
            try:
                path = request.form['delete']
                if path != 'cancel':
                    print(f'This is delete: {path}')

                    Upload.query.filter_by(path=path).delete()
                    db.session.commit()

                    img_path = os.path.join(cwd, r"static\images\\" + path)
                    os.remove(img_path)
                    flash('Plant was deleted successfully', 'danger')
                

            except:
                update = request.form['update']
                update = update.split('||')
                print(f'This is update: {update}')
                if update[0] != '':
                    info = Upload.query.filter_by(path=update[1]).first()
                    info.predict = update[0]
                    info.verified = True
                    db.session.commit()
                    flash('Plant disease category updated successfully', 'success')

            uploads = Upload.query.order_by(desc(Upload.date_upload)).all()


            return render_template('all_upload.html', total_count=all_uploads, uploads=uploads, categories=PLANT_CATEGORIES)


    return render_template('all_upload.html', total_count=all_uploads, uploads=uploads, categories=PLANT_CATEGORIES)


@app.route('/tomato', methods=['GET', 'POST'])
def tomato():
    global plant
    plant = 'tomato'
    session['plant'] = plant
    session['path'] = PATH + f'{plant}_inception_v3.h5'

    global switch,camera
    if request.method == 'POST':

        if  request.form.get('take_image') == 'Take Image':
            global check
            check = 'take_image'

        elif  request.form.get('upload') == 'Upload Image':
            check = 'upload'

        print('Value of check is:\t',check)

        if  request.form.get('start') == 'Start Camera':
            camera = cv2.VideoCapture(0)
            return render_template(f'{plant}.html',  check=check, title=plant)

        if check == 'take_image' and  request.form.get('capture') == 'Capture Image':
            global capture, what
            
            capture=1
            what = 1
            return render_template(f'{plant}.html', title=plant, check="take_image", capture=True)
            
        elif  check == 'take_image' and request.form.get('stop') == 'Stop Camera':
            camera.release()
            cv2.destroyAllWindows()

    elif request.method=='GET':
         return render_template(f'{plant}.html', title=plant)

    return render_template(f'{plant}.html',  check=check, title=plant)


@app.route('/pepper',  methods=['GET', 'POST'])
def pepper():
    global plant
    plant = 'pepper'
    session['plant'] = plant
    session['path'] = PATH + f'{plant}_inception_v3.h5'

    global switch,camera
    if request.method == 'POST':

        if  request.form.get('take_image') == 'Take Image':
            global check
            check = 'take_image'

        elif  request.form.get('upload') == 'Upload Image':
            check = 'upload'
        if  request.form.get('start') == 'Start Camera':
            camera = cv2.VideoCapture(0)
            return render_template(f'{plant}.html',  check=check, title=plant)

        if check == 'take_image' and  request.form.get('capture') == 'Capture Image':
            global capture, what
            
            capture=1
            what = 1
            return render_template(f'{plant}.html', title=plant, check="take_image", capture=True)
            
        elif  check == 'take_image' and request.form.get('stop') == 'Stop Camera':
            camera.release()
            cv2.destroyAllWindows()

    elif request.method=='GET':
         return render_template(f'{plant}.html', title=plant)

    return render_template(f'{plant}.html',  check=check, title=plant)


@app.route('/potato', methods=['GET', 'POST'])
def potato():
    global plant
    plant = 'potato'
    session['plant'] = plant
    session['path'] = PATH + f'{plant}_inception_v3.h5'
    
    global switch,camera
    if request.method == 'POST':

        if  request.form.get('take_image') == 'Take Image':
            global check
            check = 'take_image'

        elif  request.form.get('upload') == 'Upload Image':
            check = 'upload'

        if  request.form.get('start') == 'Start Camera':
            camera = cv2.VideoCapture(0)
            return render_template(f'{plant}.html',  check=check, title=plant)

        if check == 'take_image' and  request.form.get('capture') == 'Capture Image':
            global capture, what
            
            capture=1
            what = 1
            return render_template(f'{plant}.html', title=plant, check="take_image", capture=True)
            
        elif  check == 'take_image' and request.form.get('stop') == 'Stop Camera':
            camera.release()
            cv2.destroyAllWindows()

    elif request.method=='GET':
         return render_template(f'{plant}.html', title=plant)

    return render_template(f'{plant}.html',  check=check, title=plant)


@app.route('/maize', methods=['GET', 'POST'])
def maize():
    global plant
    plant = 'maize'
    session['plant'] = plant
    session['path'] = PATH + f'{plant}_inception_v3.h5'
    
    global switch,camera
    if request.method == 'POST':

        if  request.form.get('take_image') == 'Take Image':
            global check
            check = 'take_image'

        elif  request.form.get('upload') == 'Upload Image':
            check = 'upload'

        if  request.form.get('start') == 'Start Camera':
            camera = cv2.VideoCapture(0)
            return render_template(f'{plant}.html',  check=check, title=plant)

        if check == 'take_image' and  request.form.get('capture') == 'Capture Image':
            global capture, what
            
            capture=1
            what = 1
            return render_template(f'{plant}.html', title=plant, check="take_image", capture=True)
            
        elif  check == 'take_image' and request.form.get('stop') == 'Stop Camera':
            camera.release()
            cv2.destroyAllWindows()

    elif request.method=='GET':
         return render_template(f'{plant}.html', title=plant)

    return render_template(f'{plant}.html',  check=check, title=plant)


@app.route('/cassava',  methods=['GET', 'POST'])
def cassava():
    global plant
    plant = 'cassava'
    session['plant'] = plant
    session['path'] = PATH + f'{plant}_inception_v3.h5'
   
    global switch,camera
    if request.method == 'POST':

        if  request.form.get('take_image') == 'Take Image':
            global check
            check = 'take_image'

        elif  request.form.get('upload') == 'Upload Image':
            check = 'upload'

        if  request.form.get('start') == 'Start Camera':
            camera = cv2.VideoCapture(0)
            return render_template(f'{plant}.html',  check=check, title=plant)

        if check == 'take_image' and  request.form.get('capture') == 'Capture Image':
            global capture, what
            
            capture=1
            what = 1
            return render_template(f'{plant}.html', title=plant, check="take_image", capture=True)
            
        elif  check == 'take_image' and request.form.get('stop') == 'Stop Camera':
            camera.release()
            cv2.destroyAllWindows()

    elif request.method=='GET':
         return render_template(f'{plant}.html', title=plant)

    return render_template(f'{plant}.html',  check=check, title=plant)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    plant_name = session.get('plant')
    MODEL_PATH = session.get('path')
    model = load_model(MODEL_PATH, custom_objects=None, compile=True)
    stamp= str(datetime.now()).replace('-', '').replace(':', '.')

    # print('\n')
    # print(MODEL_PATH)
    if request.method == 'POST':
        # Get the file from post request
        try:
            delete = False
            f = request.files['file']
            filename = f"{plant_name}_{stamp}.png"
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', 'images', filename)
            f.save(file_path)

        except Exception:
            # print(f'\n\n\nI am here\n{capture_name}\n\n')
            delete = False
            basepath = os.path.dirname(__file__)
            file_path = os.path.join( basepath, 'static', 'images', capture_name)
            filename = capture_name


        result, label, scale = model_predict(file_path, model, plant_name, delete)
        new_upload(plant=plant_name, path=filename, predict=label, confidence=scale)
        print(f'\n\nresult is {result}')
        return result
    return None



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)

    