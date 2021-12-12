# Download the trained model
For the web demo to work, you can download 
* the trained model for tomatoes [here](https://drive.google.com/file/d/153ySl-Jc34zUOP53nnXuoxW7WC4sEKf6/view?usp=sharing). 
* the trained model for potato [here](https://drive.google.com/drive/folders/1gimk-3z4o_6vvg8pU4c3FO7xmo5Io1F4?usp=sharing)
* the trained model for pepper [here](https://drive.google.com/drive/folders/1hICgt8wGm41QeMTk-PPn4jXMK0X2wfvO?usp=sharing). my_model_checkpoint gave 96% accuracy while the second gave 90% accurary



# Change the model location
Once the trained model has been downloaded, change the model location in the `app.py` file

# Web Demo `app.py`
The flask app interface is written to test the plant diseases classifier model. To used the flask app test, change the model file location and the downloading and make sure it is specific to the plant name in `app.py`.