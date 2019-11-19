import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import requests
import time
import uuid
import base64
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

#setting image height and width
img_width, img_height = 224, 224
train_data_dir = 'enter the train directory /'
validation_data_dir = 'enter the validation test directory /'
#giving the number of training and testing samples
nb_train_samples =5232
nb_validation_samples =624
#number of epochs for which the program will run
epochs = 10
#batch size of the assigned images
batch_size = 24
#number of classes in the model
num_classes=2

#checking the image format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#making the model
model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.30))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
img_width, img_height = 224, 224

model_weights_path = 'model_saved_improved.h5'
#model = load_model(model_path)
model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    #print("image loaded")
    x = load_img(file, target_size=(img_width,img_height))
    
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    #print("prediction made by the program ",array)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: Normal")
    elif answer == 1:
	    print("Label: Pneumonia")
    return answer

def my_random_string(string_length=10):
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #print("hello going to print the file path",file_path)
            file.save(file_path)
            result = predict(file_path)
            #print(result)
            if result == 0.0:
                label='Normal'
            elif result == 1.0:
        	    label='Pneumonia'
            #print(file_path)
            filename = my_random_string(6) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='/uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {'/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='127.0.0.1',port=1800)