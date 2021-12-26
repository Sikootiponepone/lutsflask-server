import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "LUTS_Model.h5"


class _Keyword_Spotting_Service:

	model = None
	_mappings = [
		"NEGATIVE",
		"POSITIVE"
	]

	_instance = None

	def predict(self, file_path):
		
		audio, sample_rate= librosa.load(file_path, res_type='kaiser_fast')
		mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
		mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
		mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
		predicted_index=self.model.predict(mfccs_scaled_features)
		predicted_index=np.argmax(predicted_index[0])

		predicted_keyword = self._mappings[predicted_index]

		return predicted_keyword


def Keyword_Spotting_Service():
	#ensure that we only have 1 instance pf KSS
	if _Keyword_Spotting_Service._instance is None:
		_Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
		_Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
	return _Keyword_Spotting_Service._instance

if __name__ == "__main__":
	kss = Keyword_Spotting_Service()

	