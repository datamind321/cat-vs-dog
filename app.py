import streamlit as st 
from  PIL import Image, ImageOps
import cv2 
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np 
import PIL
import keras
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from keras.models import load_model
from keras.utils import load_img





def model_prediction(img,weight_file):
    model=load_model(weight_file)

    data=np.ndarray(shape=(1,224,224,3),dtype=np.float32)
    image=img 
    size=(224,224)
    image=ImageOps.fit(image,size,Image.ANTIALIAS)
    image=np.asarray(image)
    image=np.expand_dims(image,axis=0)
    output=model.predict(image) 

    if output[0][0] > output[0][1]:
        print("cat")
    else:
        print('dog')
    
    return image,output 
    
    

    # normalize_img_array=(image_array.astype(np.float32) / 255)

    

st.title('Cat Vs Dog Classifier Model')



img = st.file_uploader("",type=['jpg','png','jpeg'])
if img is not None:
    image = Image.open(img)
     
    st.image(image,use_column_width=False)
    pred=st.button("Predict")
    st.write("")
    # st.write("Classifying.....")
    if pred:
        try:
            image,output=model_prediction(image,'vgg16_1.h5') 
            if output[0][0] > output[0][1]:
                st.write("""
		# This is a Cat
			""")
    
            else:
                st.write(""" 
		# This is a Dog
			""") 
        except:
            st.write("""
		### ❗ Oops... Something Is Going Wrong
			""")  


     

    
 
    


  





  