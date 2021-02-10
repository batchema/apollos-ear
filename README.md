# Apollo's Ear: Music Classification across 13 genres
## Description
Appolo's Ear is an ensemble of three music genre classification neural networks, a nginx server, 
and a [simple react frontend]('https://www.apollosear.com'). This backend is glued together with Docker and Docker compose
and is easily replicable by following the steps below.

## Project Structure
- data_acquisition/ data acquisition and audio cleanup utility files
- classifiers.ipynb: Jupyter notebook to implement and train models
- server/ server files
- client.py: simple main file for testing

## Running the Project
### A. If interested only in implementing the models
0. Fork the repo
1. Download the [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). If you desire aquiring extra data to classify genres other
   than the ones in the dataset, check data_acquisition/audio_data_collector.py for the needed utility functions.
   Check [this tutorial](https://www.lambdatest.com/blog/selenium-webdriver-with-python/) 
   for how to use Selenium. Note that I personally used Brave Browser and that it works with the
   Chrome Driver
2. Open classifiers.ipynb in [Google Collab](https://colab.research.google.com/notebooks/intro.ipynb) (recommended) or your favorite Jupyter server
3. Follow the notebook to train and save the models. Note that if using Google Collab, you will need to link your Google Drive

### B. If interested in *A.* and also putting the dockerize project on a server
0. Follow all of ***A.***
1. Once you have trained and saved your models, put them somewhere in the server folder.
2. Edit `MLP_PATH`, `CNN_PATH`, and `RNN_PATH` in genre_prediction_service.py to point to your
MLP, CNN and RNN_LSTM models respectively. Go to Step 7 if you do not wish to test the system locally
3. Install [Docker](https://docs.docker.com/engine/install/) and 
   [Docker Compose](https://docs.docker.com/compose/install/) on your machine.
   Note that Docker Compose is included with the official installations of Docker on 
   Windows and macOS
4. In your terminal, `cd server` and run `docker-compose build`. Then, run `docker-compose up`. 
An Nginx server proxying a UWsgi server which serves server.py should now be running.
5. In client.py, make sure `URL=DOCKER_TEST_PATH` and that `TEST_PATH` points to an audio file.
6. Run client.py. It should hit your server and return a predicted genre if everything went as desired
7. Spin up a server (EC2 instance recommended as it is what I used) and scp the server/ folder onto it.
   Then make sure the http port is accessible from anywhere (On EC2, this is done by 
   [adding inbound rules](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html).
8. ssh onto the server and `cd server` then `chmod +x init.sh`
9. Run `./init.sh`. Make sure to read `init.sh` to ensure I am not making you download malware ;)
10. Copy your server ip to `SERVER_IP` in client.py, then make sure `URL=PROD_URL`. Then run the file.
    It should hit your server and return a prediction if all went well. 
    
By now, you should have 3 models for music genre classification, and a production-ready server to 
power any frontend application
    
PS: Tensorflow hangs on some servers when using the RNN_LSTM model. I am as of yet incapable of solving the bug.
If it happens to you, best course of action is to use a different model on the server. If you manage to solve the bug,
please [let me know](https://github.com/batchema/apollos-ear/issues/new).
## Special Note
I would have not been able to complete this project if not for the content of multiple 
open-source engineers. I would like to particularly highlight 
[Valerio Velardo](https://github.com/musikalkemist) 
from the [The Sound of AI](https://www.youtube.com/channel/UCZPFjMe1uRSirmSpznqvJfQ).
Anyone interested in Machine Learning for audio classification should definitely check his pages.


