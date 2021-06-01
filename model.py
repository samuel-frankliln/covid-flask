# -*- coding: utf-8 -*-
"""


@author: samuel
"""
import os
from flask import Flask,request,render_template,send_from_directory

app =Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes =["Covid","Normal"]

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/images')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        #import tensorflow as tf
        import numpy as np
        from keras.preprocessing import image

        from keras.models import load_model
        new_model = load_model('model.h5')
        new_model.summary()
        pic_dataset = image.ImageDataGenerator(
            rescale =1./255
            )
        pic_generator =pic_dataset.flow_from_directory(
            'C:/Users/samue/Desktop/covid/Dataset-20210217T030213Z-001/program/images',
            target_size =(224,224),
            batch_size =32,
            class_mode ='binary'

                )
        val= new_model.predict(pic_generator)

        if(val[0] <0.5):
            prediction=classes[0]
            
        else:
            prediction = classes[1]
        

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html",image_name=filename, text=prediction,value=val[0])

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images/images", filename)

if __name__ == "__main__":
    app.run(debug=False)