import os

import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('models/brainTumor-4category-b64e50-categorical-no-gpu.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_classname(classname):
    if classname[0][0] == 1:
        return "Glioma Tumor Present:A Glioma tumor has been detected. Gliomas, classified as primary brain tumors, originate from glial cells and can manifest in various parts of the brain. They often exhibit aggressive growth and may require a combination of treatment modalities, such as surgery, chemotherapy, and radiation therapy."
    elif classname[0][1] == 1:
        return "Meningioma Tumor Present:A Meningioma tumor has been detected. Meningiomas arise from the meninges, the protective layers covering the brain and spinal cord. These tumors are typically benign but can cause symptoms depending on their size and location. Treatment options may include observation, surgery, or radiation therapy."
    elif classname[0][2] == 1:
        return "No Tumor present:No tumor has been detected in the brain. It's important to continue monitoring for any changes in symptoms or health status, and to consult with a healthcare professional for further evaluation if necessary."
    elif classname[0][3] == 1:
        return "Pituitary Tumor Present:A Pituitary tumor has been detected. Pituitary tumors develop in the pituitary gland, a pea-sized gland at the base of the brain. Depending on the type of pituitary tumor, it may lead to hormonal imbalances and various symptoms, such as headaches, vision problems, or hormone overproduction. Treatment options range from medication to surgery, depending on the tumor's characteristics."


def getresult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/learn', methods=['GET'])
def learn():
    return render_template('learn.html')

@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')

@app.route('/logout', methods=['GET'])
def logout():
    return render_template('logout.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getresult(file_path)
        result = get_classname(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
