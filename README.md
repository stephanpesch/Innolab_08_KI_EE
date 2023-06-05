# User manual
To run the application:
1. install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Clone this [repository](https://github.com/stephanpesch/Innolab_08_KI_EE).
   ```bash
   $ git clone https://github.com/stephanpesch/Innolab_08_KI_EE.git
   $ cd Innolab_08_KI_EE
   ```
3. Create an environment, change `<env>` to whatever name you want to give your environment:
   ```bash
   $ conda create --name <env> --file requirements.txt
   $ conda activate <env>
   ```
4. To add a OpenWeatherMap Token create a file named "tokens.py" and create the variable "open_weather_map_token" like this:
   ```python
   open_weather_map_token = "token here"
   ```
   
5. To start the application:
   ```bash
   $ python main.py
   ```

## Starting page
On the starting page, there are multiple options to select from:

1. Train a model
2. Make a prediction
   1. You need to have a trained model before that
3. Get the weather forecast

## Train a model
To train a model upload a dataset with the consumption data and a data set with the temperature data on the right-hand side of the application.
After that an algorithm should be selected of one of the four algorithms on the left.
Then the “Train model” Button should be clicked.

A screen opens where the features of the dataset should be selected, on the left side you should select the time and the consumption features of the dataset, on the left side the time and temperature features.

If you want to run a grid search, tick the checkbox on the bottom of the screen. If the grid search is finished, the algorithm is executed with the hyperparameters of the grid search.
If you don’t want to run a grid search, the algorithm is executed with hardcoded hyperparameters.

The training returns a model for the forecast area and shows the result of a forecast for a test period.

## Make a prediction
If a model has been trained before, the prediction can be started from the start screen, but before pressing the button the location for the weather forecast has to be entered in the text field below.
The prediction is shown for a period of 40 hours.
## Weather forecast
If a location is entered and the weather forecast button is clicked, the weather forecast for 48 hours is shown.