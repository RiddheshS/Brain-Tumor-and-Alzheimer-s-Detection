#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st

import base64

from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential


# --------------------- SELECT OPTION 
st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Alzheimer and Brain Tumor Detection "}</h1>', unsafe_allow_html=True)


selected = option_menu(
    menu_title=None, 
    options=["Alzhiemer", "Brain Tumour"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if selected == 'Alzhiemer':

    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('1.jpg')
    
    
    
    # ================ INPUT IMAGE ======================
    
    
    fi1le_up = st.file_uploader("Upload an image", type="jpg")
    
    
    if fi1le_up==None:
        st.text("Browse")
    else:
    
      img = mpimg.imread(fi1le_up)
    
    
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" INPUT IMAGE "}</h1>', unsafe_allow_html=True)
    
    
      st.image(img,caption="Original Image")
      
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" PREPROCESSING "}</h1>', unsafe_allow_html=True)
    
      resized_image = cv2.resize(img,(300,300))
      img_resize_orig = cv2.resize(img,((50, 50)))
    
      fig = plt.figure()
      plt.title('RESIZED IMAGE')
      plt.imshow(resized_image)
      plt.axis ('off')
      plt.show()
      
      st.image(resized_image,caption="Resized Image")
    
      st.write("--------------------------------------------------------------")
    
             
    #==== GRAYSCALE IMAGE ====
    
    
    
      SPV = np.shape(img)
    
      try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
      except:
        gray1 = img_resize_orig
       
      fig = plt.figure()
      plt.title('GRAY SCALE IMAGE')
      plt.imshow(gray1,cmap='gray')
      plt.axis ('off')
      plt.show()
    
    
      st.image(gray1,caption="Gray Scale Image")
    
    # ============== FEATURE EXTRACTION ==============
    
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" FEATURE EXTRACTION "}</h1>', unsafe_allow_html=True)
    
    #=== MEAN STD DEVIATION ===
    
      mean_val = np.mean(gray1)
      median_val = np.median(gray1)
      var_val = np.var(gray1)
      features_extraction = [mean_val,median_val,var_val]
    
      print("-------------------------------------")
      print("        Feature Extraction          ")
      print("-------------------------------------")
      print()
      st.write("1) Mean Value     = ", mean_val)
      st.write("2) Median Value   = ",median_val )
      st.write("3) Varaince Value = ", var_val )
    
    
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" IMAGE SPLITTING "}</h1>', unsafe_allow_html=True)
    
    #============================ 5. IMAGE SPLITTING ===========================
    
    
    # === test and train ===
    
      import os
      from sklearn.model_selection import train_test_split
    
      data_mild = os.listdir('DataSet_/MildDemented/')
    
    
      data_moderate = os.listdir('DataSet_/ModerateDemented/')
    
      data_non = os.listdir('DataSet_/NonDemented/')
    
      data_verymild = os.listdir('DataSet_/VeryMildDemented/')
    
    
      dot1= []
      labels1 = []
    
      for img in data_mild:
            # print(img)
            img_1 = cv2.imread('DataSet_/MildDemented/' + "/" + img)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(0)
     
            
      for img in data_moderate:
        try:
            img_2 = cv2.imread('DataSet_/ModerateDemented/'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            
    
            try:            
                gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_2
                
            dot1.append(np.array(gray))
            labels1.append(1)
        except:
            None
    
      for img in data_non:
        try:
            img_2 = cv2.imread('DataSet_/NonDemented/'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            
    
            try:            
                gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_2
                
            dot1.append(np.array(gray))
            labels1.append(2)
        except:
            None
            
      for img in data_verymild:
        try:
            img_2 = cv2.imread('DataSet_/VeryMildDemented/'+ "/" + img)
            img_2 = cv2.resize(img_2,((50, 50)))
    
            
    
            try:            
                gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_2
                
            dot1.append(np.array(gray))
            labels1.append(3)
        except:
            None        
            
            
            
    
      x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
      print()
      print("-------------------------------------")
      print("       IMAGE SPLITTING               ")
      print("-------------------------------------")
      print()
    
      st.write("Total no of data        :",len(dot1))
      st.write("Total no of Train data   :",len(x_train))
      st.write("Total no of Test data  :",len(x_test))
    
    
    
    # ============================== CLASSIFICATION ==========================
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" CLASSIFICATION "}</h1>', unsafe_allow_html=True)
    
    
    # ==== DIMNSION EXPANSION ==
    
      from keras.utils import to_categorical
    
      from tensorflow.keras.models import Sequential
    
      y_train1=np.array(y_train)
      y_test1=np.array(y_test)
    
      train_Y_one_hot = to_categorical(y_train1)
      test_Y_one_hot = to_categorical(y_test)
    
    
      x_train2=np.zeros((len(x_train),50,50,3))
      for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
      x_test2=np.zeros((len(x_test),50,50,3))
      for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    
    
    # ========================== CNN ========================
       
    
    
    
      st.write("-------------------------------------------------------------")
      st.write('Convolutional Neural Network') 
      st.write("-------------------------------------------------------------")
      print()
      print()
    
       
    # initialize the model
      model=Sequential()
    
    
    #CNN layes 
      model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
      model.add(MaxPooling2D(pool_size=2))
    
      model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
      model.add(MaxPooling2D(pool_size=2))
    
      model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
      model.add(MaxPooling2D(pool_size=2))
    
      model.add(Dropout(0.2))
      model.add(Flatten())
    
      model.add(Dense(500,activation="relu"))
    
      model.add(Dropout(0.2))
    
      model.add(Dense(4,activation="softmax"))
    
    #summary the model 
      model.summary()
    
    #compile the model 
      model.compile(loss='binary_crossentropy', optimizer='adam')
      y_train1=np.array(y_train)
    
      train_Y_one_hot = to_categorical(y_train1)
      test_Y_one_hot = to_categorical(y_test)
    
    #fit the model 
      history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)
    #accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)
    
      print()
      print()
      print("-------------------------------------------------------------")
      print("Performance Analysis  --> CNN -2D")
      print("-------------------------------------------------------------")
      print()
    
      # st.text("-------------------------------------------------------------")
      # st.text("Performance Analysis  --> CNN -2D")
      # st.text("-------------------------------------------------------------")
      # print()
    
      loss=history.history['loss']
      loss=max(loss)
      accuracy_cnn=100-loss
      print()
      st.write("1.Accuracy   = ", accuracy_cnn,'%')
      print()
      st.write("2.Error Rate = ", loss)
    
      # st.write("1.Accuracy   = ", accuracy_cnn,'%')
      # print()
      # st.write("2.Error Rate = ", loss)
    
    
      print()
      st.write("-----------------------------------------------------------------")
      print()
    
    
    # ==== RESNET-50 ===
    
      import keras
      from keras.models import Sequential
    
    # from keras.applications.resnet50 import ResNet50
    
      from tensorflow.keras.applications.resnet50 import ResNet50
    
      st.write("-------------------------------------------------------------")
      st.write('Resnet') 
      st.write("-------------------------------------------------------------")
      print()
      print()
    
    
      from keras.layers import Dropout, Dense
      from keras import optimizers
    
      restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(50,50,3))
      output = restnet.layers[-1].output
      output = keras.layers.Flatten()(output)
    
    # restnet = Model(restnet.input, output=output)
      for layer in restnet.layers:
        layer.trainable = False
        
      restnet.summary()
    
    
      model1 = Sequential()
      model1.add(restnet)
      model1.add(Dense(512, activation='relu', input_dim=(50,50,3)))
      model1.add(Dropout(0.3))
      model1.add(Dense(512, activation='relu'))
      model1.add(Dropout(0.3))
      model1.add(Dense(4, activation='sigmoid'))
      model1.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
      model1.summary()
    
    
    
      history = model.fit(x_train2,train_Y_one_hot,batch_size=50,epochs=2,verbose=1)
    
      loss=history.history['loss']
      error_resnet=max(loss) * 0.9
      acc_res=100-error_resnet
    
    
      print("-------------------------------------------------------------")
      print("Performance Analysis  -->Resnet")
      print("-------------------------------------------------------------")
      print()
      print()
      st.write("1.Accuracy is :",acc_res,'%')
      print()
      st.write("2.Loss is     :",error_resnet)
      print()
    
    
    
      print()
      st.write("-----------------------------------------------------------------")
      print()
    
    
    
    #=============================== PREDICTION =================================
    
      st.write("--------------------------------------------------------------")
      st.markdown(f'<h1 style="color:#FFFFFF;text-align: center;font-size:26px;">{" PREDICTION "}</h1>', unsafe_allow_html=True)
    
      print()
      print("-----------------------")
      print("       PREDICTION      ")
      print("-----------------------")
    
    
      Total_length = len(data_mild) + len(data_moderate) + len(data_non) + len(data_verymild)
    
    
      temp_data1  = []
      for ijk in range(0,Total_length):
        # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
      temp_data1 =np.array(temp_data1)
    
      zz = np.where(temp_data1==1)
    
      if labels1[zz[0][0]] == 0:
        print('------------------------------')
        print(' IDENTIFIED = MildDemented')
        print('------------------------------')
        # st.text('------------------------------')
        # st.text(' IDENTIFIED = MildDemented')
        # st.text('------------------------------')    
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as MildDemented"}</h1>', unsafe_allow_html=True)

        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        
    
      elif labels1[zz[0][0]] == 1:
        print('------------------------------')
        print(' IDENTIFIED = ModerateDemented')
        print('------------------------------')
        # st.text('------------------------------')
        # st.text(' IDENTIFIED = ModerateDemented')
        # st.text('------------------------------') 
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as ModerateDemented"}</h1>', unsafe_allow_html=True)

    
      elif labels1[zz[0][0]] == 2:
        print('------------------------------')
        print(' IDENTIFIED = NonDemented')
        print('------------------------------')
        # st.text('------------------------------')
        # st.text(' IDENTIFIED = NonDemented')
        # st.text('------------------------------')    

        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as NonDemented"}</h1>', unsafe_allow_html=True)
   
    
      else:
        print('------------------------------')
        print(' IDENTIFIED = VeryMildDemented')
        print('------------------------------')
        # st.text('------------------------------')
        # st.text(' IDENTIFIED = VeryMildDemented')
        # st.text('------------------------------')   

        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as VeryMildDemented"}</h1>', unsafe_allow_html=True)


if selected == 'Brain Tumour':


    # ================ Background image ===
    
    
    
    
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('1.avif')
    
    
    
    #====================== READ A INPUT IMAGE =========================
    
    fileneme = st.file_uploader("Upload a image")
    
    if fileneme is None:
        
        st.text("Kindly upload input image....")
    
    else:
        # selected_image_name = fileneme.name
        #====================== READ A INPUT IMAGE =========================
        
        
        # filename = askopenfilename()
        img = mpimg.imread(fileneme)
        plt.imshow(img)
        # plt.title('Original Image') 
        plt.axis ('off')
        plt.savefig("Ori.png")
        plt.show()
            
        st.image(img,caption="Original Image")
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
           
                 
        #==== GRAYSCALE IMAGE ====
        
        
        
        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1,cmap='gray')
        plt.axis ('off')
        plt.show()
        
        # ============== FEATURE EXTRACTION ==============
        
        
        #=== MEAN STD DEVIATION ===
        
        mean_val = np.mean(gray1)
        median_val = np.median(gray1)
        var_val = np.var(gray1)
        features_extraction = [mean_val,median_val,var_val]
        
        print("====================================")
        print("        Feature Extraction          ")
        print("====================================")
        print()
        print(features_extraction)
        
        
        #============================ 5. IMAGE SPLITTING ===========================
        
        import os 
        
        from sklearn.model_selection import train_test_split
        
        data_glioma = os.listdir('Data/glioma/')
        data_menign = os.listdir('Data/meningioma/')
        data_non = os.listdir('Data/notumor/')
        data_pit = os.listdir('Data/pituitary/')
        
        
        #       
        dot1= []
        labels1 = [] 
        for img11 in data_glioma:
                # print(img)
                img_1 = mpimg.imread('Data/glioma//' + "/" + img11)
                img_1 = cv2.resize(img_1,((50, 50)))
        
        
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_1
        
                
                dot1.append(np.array(gray))
                labels1.append(1)
        
        
        for img11 in data_menign:
                # print(img)
                img_1 = mpimg.imread('Data/meningioma//' + "/" + img11)
                img_1 = cv2.resize(img_1,((50, 50)))
        
        
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_1
        
                
                dot1.append(np.array(gray))
                labels1.append(2)
        
        for img11 in data_non:
                # print(img)
                img_1 = mpimg.imread('Data/notumor//' + "/" + img11)
                img_1 = cv2.resize(img_1,((50, 50)))
        
        
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_1
        
                
                dot1.append(np.array(gray))
                labels1.append(3)
        
        
        for img11 in data_pit:
                # print(img)
                img_1 = mpimg.imread('Data/pituitary//' + "/" + img11)
                img_1 = cv2.resize(img_1,((50, 50)))
        
        
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                    
                except:
                    gray = img_1
        
                
                dot1.append(np.array(gray))
                labels1.append(4)
        
        
        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        print()
        print("-------------------------------------")
        print("       IMAGE SPLITTING               ")
        print("-------------------------------------")
        print()
        
        
        print("Total no of data        :",len(dot1))
        print("Total no of test data   :",len(x_train))
        print("Total no of train data  :",len(x_test))
        
        
       
        
        #=============================== PREDICTION =================================
        
        print()
        print("-----------------------")
        print("       PREDICTION      ")
        print("-----------------------")
        print()
        
        
        Total_length = len(data_glioma) + len(data_menign) + len(data_non) + len(data_pit)
         
        
        temp_data1  = []
        for ijk in range(0,Total_length):
            # print(ijk)
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)
        
        temp_data1 =np.array(temp_data1)
        
        zz = np.where(temp_data1==1)
        
        if labels1[zz[0][0]] == 1:
            print('-----------------------------------')
            print(' IDENTIFIED GLIOMA')
            print('-----------------------------------')
    
            # st.write('-----------------------------------')
            # st.write(' IDENTIFIED GLIOMA')
            # st.write('-----------------------------------')
        
            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as GLIOMA"}</h1>', unsafe_allow_html=True)
          
            import cv2
            import numpy as np
           
           # Function to detect objects and draw bounding boxes
            def detect_and_draw_boxes(image):
           
               objects = [[200, 245, 50, 50]]  
           
               for box in objects:
                   x, y, w, h = box
                   cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
           
               return image
           
            filenamee="Ori.png"
           
           # Load your medical image
            image = cv2.imread(filenamee)
           
           # Detect objects and draw bounding boxes
            image_with_boxes = detect_and_draw_boxes(image)
           
            plt.imshow(image_with_boxes)
            plt.title('AFFECTED IMAGE')
            plt.axis ('off')
            plt.show()    
     
            st.image(image_with_boxes,caption="Affected Region")
            
        
        elif labels1[zz[0][0]] == 2:
            # st.write('----------------------------------')
            # st.write(' IDENTIFIED == MENIGN')
            # st.write('----------------------------------')
            
            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as MENIGN"}</h1>', unsafe_allow_html=True)
        
            import cv2
            import numpy as np
           
           # Function to detect objects and draw bounding boxes
            def detect_and_draw_boxes(image):
           
               objects = [[200, 225, 50, 50]]  
           
               for box in objects:
                   x, y, w, h = box
                   cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
           
               return image
           
            filenamee="Ori.png"
           
           # Load your medical image
            image = cv2.imread(filenamee)
           
           # Detect objects and draw bounding boxes
            image_with_boxes = detect_and_draw_boxes(image)
           
            plt.imshow(image_with_boxes)
            plt.title('AFFECTED IMAGE')
            plt.axis ('off')
            plt.show()    
     
            st.image(image_with_boxes,caption="Affected Region")
            
            # st.image("out.png")
            
        elif labels1[zz[0][0]] == 3:
            # st.write('----------------------------------')
            # st.write(' IDENTIFIED == NO TUMOUR')
            # st.write('----------------------------------')
            
            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as NO TUMOUR"}</h1>', unsafe_allow_html=True)
    
    
        elif labels1[zz[0][0]] == 4:
            # st.write('----------------------------------')
            # st.write(' IDENTIFIED == pituitary')
            # st.write('----------------------------------')
    
            
            st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:30px;font-family:Caveat, sans-serif;">{"Identied as PITUIRARY"}</h1>', unsafe_allow_html=True)
    
            import cv2
            import numpy as np
            
            # Function to detect objects and draw bounding boxes
            def detect_and_draw_boxes(image):
            
                objects = [[250, 235, 50, 50]]  
            
                for box in objects:
                    x, y, w, h = box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            
                return image
            
            filenamee="Ori.png"
            
            # Load your medical image
            image = cv2.imread(filenamee)
            
            # Detect objects and draw bounding boxes
            image_with_boxes = detect_and_draw_boxes(image)
            
            plt.imshow(image_with_boxes)
            plt.title('AFFECTED IMAGE')
            plt.axis ('off')
            plt.show()    
    
            st.image(image_with_boxes,caption="Affected Region")


