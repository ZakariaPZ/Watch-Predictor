from flask import Flask, request, jsonify
import numpy as np 
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model-weights/xception.h5'
model = load_model(MODEL_PATH)

CLASS_DICT = {
    0: 'Cartier',
    1: 'Omega',
    2: 'Rolex',
    3: 'Seiko'
}

# model.make_predict_function()

def load_img(input_image, shape):
    img = Image.open(input_image).convert('RGB')
    img = img.resize((shape, shape))
    img = image.img_to_array(img)
    return np.reshape(img, [1, shape, shape, 3])/255

#This route accepts post requests 

@app.route('/classify', methods=["POST", "OPTIONS"])
def classify():
	files = request.files
	print(files)
	pred_img = load_img(files['image'], 224)
	pred = model.predict(pred_img)
	# pred = CLASS_DICT[np.argmax(model.predict(pred_img))]
	print(pred)
	return {
		"brandPredictions": sorted(
			list(
				zip(

			list(CLASS_DICT.values()),
			[round(prediction, 4) for prediction in map(float, pred[0])]

				)
			),
		key=lambda p: p[1],
		reverse=True
		)
	}

if __name__=='__main__':
	app.run(host="0.0.0.0", port=8000, debug=True)

