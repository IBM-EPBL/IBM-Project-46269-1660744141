import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app = Flask(__name__) #initializing the flask object in the variable name as app
@app.errorhandler(404)
def not_found(e):
  
# defining function
  return render_template("Classify.html")
model = load_model("Nutrition Analyzer.h5")

@app.route('/' ,methods=['GET']) #routing the html

def home():
    return render_template('Home.html')  #displayed on the html page


@app.route('/predict',methods=['GET','POST'])
def upload():
        if request.method=='POST':
            f = request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            f.save(filepath)
            img = image.load_img(filepath,target_size=(64,64))
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis=0)
            pred = np.argmax(model.predict(x),axis=1)
            index = ["Apple","Banana","Orange","Pineapple","Watermelon"]
            result = str(index[pred[0]]) 
            print(result)
            x=result
            if(result == "Apple"):
                content = "Sugar: 10.3g,\nFiber: 2.4g, Serving size: 100g, Sodium: 1mg, Potassium: 11mg, Fat saturated: 0g,   Fat total: 0.2g, Calories: 53g, Cholestrol: 0mg, Protein: 3g, Carbohydrates total: 14.1g."
            elif(result == "Banana"):
                content = "Sugar: 12.3g,\nFiber: 2.6g, Serving size: 100g, Sodium: 1mg, Potassium: 22mg, Fat saturated: 0.1g,   Fat total: 0.3g, Calories: 89.4g, Cholestrol: 0mg, Protein: 1.1g, Carbohydrates total: 23.2g."
            elif(result == "Orange"):
                content = "Sugar: 16.8g,\nFiber: 4.3g, Serving size: 100g, Sodium: 1mg, Potassium: 22mg, Fat saturated: 0g,   Fat total: 0.2g, Calories: 84g, Cholestrol: 0mg, Protein: 1.7g, Carbohydrates total: 21.2g."
            elif(result == "Pineapple"):
                content = "Sugar: 9.9g,\nFiber: 21.4g, Serving size: 100g, Sodium: 0mg, Potassium: 8mg, Fat saturated: 0g,   Fat total: 0.1g, Calories: 50.8g, Cholestrol: 0mg, Protein: 10.5g, Carbohydrates total: 13g."
            elif(result == "Watermelon"):
                content = "Sugar: 16.3g,\nFiber: 2.3g, Serving size: 100g, Sodium: 1mg, Potassium: 22mg, Fat saturated: 0g,   Fat total: 0.3g, Calories: 83g, Cholestrol: 0mg, Protein: 0.9g, Carbohydrates total: 21.6g."
            else:
                content = ""
            return render_template("Result.html",showcase=(content),showcase1=(x))
if __name__=="__main__": 
    app.run(debug = False) #running the app
