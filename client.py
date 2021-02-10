import requests

SERVER_IP = ''  # Server IP
PROD_URL = f'https://{SERVER_IP}/predict'  # Server URL
DOCKER_TEST_URL = 'http://127.0.0.1/predict'  # Localhost with Docker or Nginx (port 80)
TEST_URL = 'http://127.0.0.1:5000/predict'  # Localhost
TEST_PATH = ''  # Test audio file

URL = PROD_URL

if __name__ == "__main__":
    song = open(TEST_PATH, 'rb')
    req_data = {'file': (TEST_PATH, song, 'audio/wav')}
    response = requests.post(URL, files=req_data)
    # print(response)
    res_data = response.json()

    print(f"Predicted Genre: {res_data['text']}")
    print(res_data)
