import subprocess
import magic
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import statistics
from enum import Enum

"""Constants (from training)"""
dirname = os.path.dirname(__file__)
MLP_PATH = os.path.join(dirname, 'saved_models/mlp_model0')
CNN_PATH = os.path.join(dirname, 'saved_models/cnn_model0')
RNN_PATH = os.path.join(dirname, 'saved_models/rnn_model0')
MODEL_PATH = MLP_PATH

MAPPINGS = {
    0: 'rumba', 1: 'disco', 2: 'rock', 3: 'classical', 4: 'metal',
    5: 'afrobeat', 6: 'pop', 7: 'reggae', 8: 'blues',
    9: 'coupe_decale', 10: 'hiphop', 11: 'jazz', 12: 'country'
}

SAMPLE_RATE = 22050
INTERVAL_LENGTH = 30
SAMPLES_PER_INTERVAL = int(INTERVAL_LENGTH * SAMPLE_RATE / 10)
NUM_MFCC_VECTORS_PER_SEGMENT = 130
HOP_LENGTH = 512
N_MFCC = 13
N_FFT = 2048
"""***********************************************"""


def get_duration(filepath):
    """Get audio file duration in seconds
    Arguments:
        filepath: String representation of path to audio file
    """
    # ffprobe = f'ffprobe -i {filepath} -show_format -v quiet | sed -n \'s/duration=//p\''
    # ffprobe = f'ffprobe -i {filepath} -show_entries format=duration -v quiet -of csv="p=0"'
    ffprobe = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {filepath}'
    cmd = subprocess.run(ffprobe, shell=True, stdout=subprocess.PIPE)
    res = cmd.stdout
    print(res)
    return int(float(res))


def remove_silence(filepath):
    """Remove silence from audio file
    Arguments:
        filepath: String representation of path to audio file
    """
    folder = os.path.dirname(filepath)
    if not folder:
        temp = 'temp.wav'
    else:
        temp = f'{folder}/temp.wav'
    ffmpeg = f'ffmpeg -i {filepath} -af silenceremove=1:0:-75dB {temp}'
    subprocess.run(ffmpeg, shell=True)
    subprocess.run(f'mv -f {temp} {filepath}', shell=True)


def extract_features_mlp(signal, sample_rate, start, finish):
    """
    Extract feature vectors of signal between start and finish interval
    limits for MLP classifier
    Arguments:
        signal: Librosa-extracted signal
        sample_rate: Librosa-extracted sample rate
        start: start of interval
        finish: End of interval
    Returns: Features Vector for MLP classifier
    """
    # Extract spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=signal[start:finish],
                                                  sr=sample_rate, n_fft=N_MFCC, hop_length=HOP_LENGTH)
    # Extract spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=signal[start:finish],
                                                 sr=sample_rate, n_fft=N_MFCC, hop_length=HOP_LENGTH)

    # Extract spectral rolloff
    spec_ro = librosa.feature.spectral_rolloff(y=signal[start:finish],
                                               sr=sample_rate, hop_length=HOP_LENGTH)

    # Extract chromagram
    chroma_stft = librosa.feature.chroma_stft(y=signal[start:finish],
                                              sr=sample_rate, hop_length=HOP_LENGTH)

    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=signal[start:finish],
                                             hop_length=HOP_LENGTH)

    # Extract tempogram
    tpg = librosa.feature.tempogram(y=signal[start:finish],
                                    sr=sample_rate, hop_length=HOP_LENGTH)

    # Start row of data for line segment
    row = [np.mean(spec_cent), np.mean(spec_bw), np.mean(spec_ro), np.mean(chroma_stft), np.mean(zcr), np.mean(tpg)]

    # Extract mfcc and take the mean of every coefficient
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc = np.mean(mfcc, axis=1)

    # final array of features
    row = np.array(row)
    row = np.concatenate([row, mfcc])

    return row


def extract_features_cnn(signal, sample_rate, start, finish):
    """
    Extract feature vectors of signal between start and finish interval
    limits for CNN classifier
    Arguments:
        signal: Librosa-extracted signal
        sample_rate: Librosa-extracted sample rate
        start: start of interval
        finish: End of interval
    Returns: Features Vector for CNN classifier
    """
    spec = librosa.feature.melspectrogram(signal[start:finish],
                                          sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = spec.T
    return spec[:130, :]


class Classifiers(Enum):
    """
    Enum class to indicate which classifier to use
    """
    MLP = 0
    CNN = 1
    RNN = 2


# Raise when file is not an audio file
class FileNotAudioError(Exception):
    pass


# Raise when audio length is too short (default 30 seconds)
class AudioTooShortError(Exception):
    pass


class _GenrePredictionService:
    """Singleton Class for audio music genre prediction"""
    _instance = None
    model = None
    classifier = Classifiers.MLP  # Default classifier

    def predict_genre(self, filepath):
        """Predict genre of a song given
        Arguments:
            filepath --  String representation of the path to the file
        """
        # Restrict fileType
        if magic.from_file(filepath, mime=True).split('/')[0] != 'audio':
            raise FileNotAudioError('Submitted file must be an audio file. Consider using .wav file')

        samples_features = self.collect_features(filepath)

        if samples_features is None:
            return None
        if len(samples_features) == 0:
            return None
        predictions = []
        print("predicting song samples...")
        for sample in samples_features:
            prediction = np.argmax(self.model.predict(np.expand_dims(sample, axis=0)))
            predictions.append(prediction)
        print("samples predictions done")
        prediction = statistics.mode(predictions)
        return MAPPINGS[prediction]

    def collect_features(self, filepath):
        """Collect features per samples of input audio file
        Arguments:
            filepath: String representation of path to the file
        Return:
            Numpy array of mfcc feature vectors
        """
        # Make sure the file is a wav file
        # Convert if necessary
        temp = "input.wav"
        remove_temp = False
        if filepath[-4:] != ".wav":
            ffmpeg = f'ffmpeg -y -i {filepath} -f wav {temp}'
            subprocess.run(ffmpeg, shell=True)
            remove_temp = True
        else:
            temp = filepath
        print("removing silence...")
        remove_silence(temp)
        print("silence removal done")
        print("getting song duration...")
        duration = get_duration(temp)
        print(f"song duration got: {duration}")
        print("getting features...")
        if duration < 30:
            raise AudioTooShortError('non-silent audio less than 30 seconds')
        features = self.collect_features_in_intervals(temp, duration)
        print("features acquisition done")
        print("features acquisition done")
        print("removing temp files...")
        if remove_temp:
            os.remove(temp)
        print("temp files removed...")
        return np.array(features)

    def collect_features_in_intervals(self, filepath, duration):
        """Collect features of samples in 30 second intervals
        Arguments:
            filepath: String representation of path to the file
            duration: duration of the song
        Returns: list of features collected in 30 seconds intervals (1 or 3) gathered
            at the beginning of the song
        """
        features = []
        # Define constants used in shaping training features
        signal, sample_rate = librosa.load(filepath, sr=SAMPLE_RATE)
        # Get 10, 20, 30 3-second samples depending on length of song
        if duration < 30:
            raise AudioTooShortError('audio is less than 30 seconds in length')
        elif duration < 60:
            features = self.get_interval_features(features, signal, sample_rate, 0, 10)
        elif duration < 90:
            for i in range(0, 2):
                start = SAMPLES_PER_INTERVAL * i * 10
                limit = 10 * (i + 1)
                features = self.get_interval_features(features, signal, sample_rate, start, limit)
        else:
            for i in range(0, 3):
                start = SAMPLES_PER_INTERVAL * i * 10
                limit = 10 * (i + 1)
                features = self.get_interval_features(features, signal, sample_rate, start, limit)

        return features

    def get_interval_features(self, collector, signal, sample_rate, start, limit):
        """Get mfcc vectors for a 30 second interval
        Arguments:
            collector: A list in which to append the vectors
            signal: Librosa-extracted signal
            sample_rate: Librosa-extracted sample rate
            start: Integer start point of the interval
            limit: limit on the length of the collector
        Returns: list of features collected in a 30-second interval
            starting at `start`
        """
        if self.classifier == Classifiers.CNN:
            specs = np.empty((0, 130, 128))
            for d in range(10):
                if len(collector) == limit:
                    break
                temp_start = start + SAMPLES_PER_INTERVAL * d
                finish = temp_start + SAMPLES_PER_INTERVAL
                spect = extract_features_cnn(signal, sample_rate, temp_start, finish)
                specs = np.append(specs, [spect], axis=0)
            specs = specs[..., np.newaxis]
            if len(collector) > 0:
                specs = np.append(collector, specs, axis=0)
            return specs

        for d in range(10):
            if len(collector) == limit:
                break
            temp_start = start + SAMPLES_PER_INTERVAL * d
            finish = temp_start + SAMPLES_PER_INTERVAL
            features = self.extract_features(signal, sample_rate, temp_start, finish)
            if self.classifier == Classifiers.RNN:
                if len(features) == NUM_MFCC_VECTORS_PER_SEGMENT:
                    collector.append(features.tolist())
            else:
                collector.append(features.tolist())

        return collector

    def extract_features(self, signal, sample_rate, start, finish):
        """ Extract feature vectors of signal at particular interval
        Arguments:
            signal: Librosa-extracted signal
            sample_rate: Librosa-extracted sample rate
            start: start of interval
            finish: End of interval
        Returns: Feature Vectors for _classifier
        """
        if self.classifier == Classifiers.MLP:
            return extract_features_mlp(signal, sample_rate, start, finish)
        elif self.classifier == Classifiers.RNN:
            return librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT,
                                        hop_length=HOP_LENGTH).T.tolist()


def Genre_Prediction_Service(classifier=Classifiers.MLP):
    """
    Create or access GenrePredictionService instance
    Arguments:
        classifier: instance of Classifiers. Default is Classifiers.MLP
    """
    # Ensure singleton service
    if _GenrePredictionService._instance is None:
        _GenrePredictionService._instance = _GenrePredictionService()
    elif classifier == _GenrePredictionService._instance.classifier:
        return _GenrePredictionService._instance

    # Load appropriate model
    model_path = MLP_PATH
    if classifier == Classifiers.CNN:
        model_path = CNN_PATH
    elif model_path == Classifiers.RNN:
        model_path = RNN_PATH
    _GenrePredictionService.model = load_model(model_path)
    _GenrePredictionService.classifier = classifier

    return _GenrePredictionService._instance


if __name__ == '__main__':
    filetype = magic.from_file('../../data_acquisition/data/hiphop/hip_hop0000.wav', mime=True)
    # print(filetype)
    # predictor = Genre_Prediction_Service(classifier=Classifiers.CNN)
    # print(predictor.predict_genre('../../data_acquisition/data/coupe_decale/coupe_decale00019.wav'))
    # print(predictor.predict_genre('../../data_acquisition/data/afrobeat/afrobeat00012.wav'))
    # print(predictor.predict_genre('../../data_acquisition/data/hiphop/hip_hop0000.wav'))

    # Put this in article for this:
    # I take multiple samples and then take the mode because statistically if I classify 10 different snippets,
    # 8 should be the right genre. So, the mode is likely to give me the right answer.
    # Assuming that the predictions for each genre is an independent event, the likelyhood that the correct genre
    # is not selected as the mode is (1-.8)^(n/2) where n is the number of segments
