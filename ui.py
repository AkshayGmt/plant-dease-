from flask import Flask, request, render_template
from PIL import Image
import io
from io import BytesIO

app = Flask(__name__)
@app.route('/')
def home():   
    message="11"
    return render_template('home.html',statusCode='Success',message=message)

@app.route('/upload',methods=['POST', 'GET'])
def upload():   
   
    
        # Get the uploaded image from the request
        image1 = request.files['image']

        # Load the image using Pillow
        img1 = image = Image.open(image1)

        
        import numpy as np # linear algebra
        import pandas as pd

        # Importing Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Convolution2D
        from keras.layers import MaxPooling2D
        from keras.layers import Flatten
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import BatchNormalization

        # Initializing the CNN
        classifier = Sequential()

        # Convolution Step 1
        classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

        # Max Pooling Step 1
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        classifier.add(BatchNormalization())

        # Convolution Step 2
        classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

        # Max Pooling Step 2

        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
        classifier.add(BatchNormalization())

        # Convolution Step 3
        classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        classifier.add(BatchNormalization())

        # Convolution Step 4
        classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
        classifier.add(BatchNormalization())

        # Convolution Step 5
        classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

        # Max Pooling Step 3
        classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
        classifier.add(BatchNormalization())

        # Flattening Step
        classifier.add(Flatten())

        # Full Connection Step
        classifier.add(Dense(units = 4096, activation = 'relu'))
        classifier.add(Dropout(0.4))
        classifier.add(BatchNormalization())
        classifier.add(Dense(units = 4096, activation = 'relu'))
        classifier.add(Dropout(0.4))
        classifier.add(BatchNormalization())
        classifier.add(Dense(units = 1000, activation = 'relu'))
        classifier.add(Dropout(0.2))
        classifier.add(BatchNormalization())
        classifier.add(Dense(units = 38, activation = 'softmax'))
        classifier.summary()

        classifier.load_weights('AlexNetModel.hdf5')

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        from keras import layers
        for i, layer in enumerate(classifier.layers):
            print(i, layer.name)


        # we chose to train the top 2 conv blocks, i.e. we will freeze
        # the first 8 layers and unfreeze the rest:
        print("Freezed layers:")
        for i, layer in enumerate(classifier.layers[:20]):
            print(i, layer.name)
            layer.trainable = False


        #trainable parameters decrease after freezing some bottom layers   
        classifier.summary()


        from keras import optimizers
        classifier.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.005),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        # image preprocessing
        from keras.preprocessing.image import ImageDataGenerator

        train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        fill_mode='nearest')

        valid_datagen = ImageDataGenerator(rescale=1./255)

        batch_size = 128
        #base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

        training_set = train_datagen.flow_from_directory('train/',
                                                        target_size=(224, 224),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

        valid_set = valid_datagen.flow_from_directory('val/',
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

        class_dict = training_set.class_indices
        print(class_dict)

        li = list(class_dict.keys())
        print(li)

        train_num = training_set.samples
        valid_num = valid_set.samples
        # checkpoint
        from keras.callbacks import ModelCheckpoint
        weightpath = "best_weights_9.hdf5"
        checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        callbacks_list = [checkpoint]

        #fitting images to CNN
        history = classifier.fit_generator(training_set,
                                steps_per_epoch=train_num//batch_size,
                                validation_data=valid_set,
                                epochs=1,
                                validation_steps=valid_num//batch_size,
                                callbacks=callbacks_list)
        #saving model
        filepath="AlexNetModel.hdf5"
        classifier.save(filepath)


        print(history.history.keys())

        #plotting training values
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        print(acc)

        # predicting an image
        import keras.utils as image
        import numpy as np
        #image_path = "C:/Users/HP/Desktop/test/test/1.JPG"
       # new_img = image.load_img(Image, target_size=(224, 224))
        img= img1.resize((224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255

        print("Following is our prediction:")
        prediction = classifier.predict(img)
        print(prediction)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        d = prediction.flatten()
        j = d.max()
        for index,item in enumerate(d):
            if item == j:
                class_name = li[index]

        ##Another way
        img_class = np.argmax(classifier.predict(img),axis=1)
        img_prob =np.argmax(classifier.predict(img),axis=1)
        print(img_class ,img_prob )



        print(class_name)

        img_data = BytesIO()
        img1.save(img_data, format='PNG')
        img_data.seek(0)
        import base64
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')
       
        

            # render the HTML template with the matching CSV contents
            #return render_template('upload.html',img=img_base64,message=class_name)

        # if it's a GET request, just render the template without any CSV contents
        return render_template('upload.html',img=img_base64, message=class_name)


       
        #return render_template('upload.html',statusCode='Success',message=a,img=img_base64)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
