import os
from os import path
import sys
import random
from flask import Flask, request, jsonify
from genre_prediction_service import Genre_Prediction_Service
from genre_prediction_service import Classifiers
from genre_prediction_service import AudioTooShortError
from genre_prediction_service import FileNotAudioError
# from flask_cors import CORS # uncomment this during testing if not using a prod server like Nginx

sys.path.append(path.join(path.dirname(__file__), '../..'))
app = Flask(__name__)
# CORS(app) # uncomment this during testing if not using a prod server like Nginx
REVIEWS_LOG_PATH = 'logs/reviews_log.txt'
EXCEPTIONS_LOG_PATH = 'logs/exception_logs.txt'
LINKS = {
    "afrobeat": "https://en.wikipedia.org/wiki/Afrobeats",
    "blues": "https://en.wikipedia.org/wiki/Blues",
    "classical": "https://en.wikipedia.org/wiki/Classical_music",
    "country": "https://en.wikipedia.org/wiki/Country_music",
    "coupe_decale": "https://en.wikipedia.org/wiki/Coup%C3%A9-d%C3%A9cal%C3%A9",
    "disco": "https://en.wikipedia.org/wiki/Disco",
    "hiphop": "https://en.wikipedia.org/wiki/Hip_hop",
    "jazz": "https://en.wikipedia.org/wiki/Jazz",
    "metal": "https://en.wikipedia.org/wiki/Heavy_metal_music",
    "pop": "https://en.wikipedia.org/wiki/Pop_music",
    "reggae": "https://en.wikipedia.org/wiki/Reggae",
    "rock": "https://en.wikipedia.org/wiki/Rock_music",
    "rumba": "https://en.wikipedia.org/wiki/Rumba"
}


def _predict(audio_file):
    """
    Helper function for the /predict route that actually
    calls the prediction functions
    Arguments:
        audio_file: The audio file to be predicted
    """
    # Save audio file to temp file
    filename = audio_file.filename.split("/")[-1]
    local_path = f'{str(random.randint(0, 100000))}{filename}'
    audio_file.save(local_path)

    # Make Prediction
    print("initializing genre prediction service...")
    GPS = Genre_Prediction_Service(classifier=Classifiers.CNN)
    print("genre prediction initialization done")
    try:
        prediction = GPS.predict_genre(local_path)
    except Exception as e:
        # Delete temp file in case of exception
        os.remove(local_path)
        # Log some information about the exception
        with open(EXCEPTIONS_LOG_PATH, 'a') as log:
            log.write(f'filename: {filename}, exception: {e.__class__.__name__}\n')
        raise e

    # Delete File after prediction
    os.remove(local_path)

    return prediction


@app.route("/predict", methods=["POST"])
def predict():
    """
    Flask route for prediction
    """
    # Get audio file and save it
    audio_file = request.files["file"]

    # Get prediction
    try:
        prediction = _predict(audio_file)
        if prediction is None:
            return jsonify({'success': False}), 512
        else:
            key, text = prediction, prediction
            if prediction == 'coupe_decale':
                text = 'coupé décalé'
            elif prediction == 'hiphop':
                text = 'hip hop'

            ret = {
                'link': LINKS[key], 'text': text, 'success': True
            }
        return jsonify(ret)
    except AudioTooShortError:
        return jsonify({'success': False}), 419

    except FileNotAudioError:
        return jsonify({'success': False}), 416


# Test
if __name__ == "__main__":
    app.run(debug=True)
