import random
import os
from flask import Flask, request, jsonify, render_template, flash
from keyword_spotting_service2 import Keyword_Spotting_Service

app = Flask(__name__)
app.secret_key="Sikooti"

@app.route('/',methods=['GET'])
def hello_world():
	flash("Welcome !")
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

	#get audio file and save it
	audio_file = request.files["audiofile"]
	file_name = str(random.randint(0,100000)) 
	audio_file.save(file_name)

	#invoke keyword spotting service
	kss = Keyword_Spotting_Service()

	#make a prediction
	predicted_keyword = kss.predict(file_name)

	#remove the audio file
	os.remove(file_name)

	#send back the predicted key word in json format
	data = {"result": predicted_keyword}

	flash("You are: " + str(predicted_keyword))

	return render_template('index.html', prediction=jsonify(data))



if __name__=="__main__":
	app.run(port=3000, debug=True)
