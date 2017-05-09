# Synopsis

This is a forecasting and analytics engine.
Given certain assumptions on the existence, and quality of data, it provides:

* Building BMS beginning of day rampup recommendation, with confidence intervals 
  (given by 'left_bound' and 'right_bound').
* Building electric demand, steam demand, water consumption, and
  occupancy predictions, with confidence intervals (given by 'left_bound' and
  'right_bound').

# Installation

* [Install the Python 2.7 version of 
Anaconda 4.0.0 (64-bit)](https://www.continuum.io/downloads).
* Download the desired release of the analytics suite 
to your local hard drive. If you have a local copy of git, this can be
done by running `git clone https://github.com/dkarapetyan/larkin` in a unix
shell.
* From a unix shell, run `pip -e install $PROJECT_ROOT`. After installation,
do not move the `$PROJECT_ROOT` directory, as this will break the
installation.

# Execution

* Once installation is successful, execute `run_analytics` in a bash shell 
  (it is automatically added to your `PATH` environment variable 
  by the installation process). This is the entry point for the analytics
  suite.

* The user will need to add the following variables to the shell environment from which
  the suite is run: 
  WUND_URL, DB_HOST, DB_PORT, DB_SOURCE, DB_USERNAME, and DB_PASSWORD.
  
* Please make sure to copy over the test weather database (history and forecasts
  tables) currently being used
  by analytics. We have built an archive of forecast data that is required for
  the feature of running previous predictions to work properly.

# Options and Features

* For options and features, please execute `run_analytics -h`
  in your shell. 
 
* If a bms prediction time does not exist for the building, or can't be 
  computed from the available data, a sentinel
  value of "2200-01-01 00:00:00+0000", representing 'infinity', 
  will be outputted by the model. 


# Scheduling on the Cloud

* After installation, please setup a scheduler to
execute `run_analytics --weather_update`
every 15 minutes, in order to continue adding weather data to the historical
and forecast tables. Failing to do so may result in the failure of 
the 'previous prediction' feature for certain dates.

* Given the current amount of data, the model takes a maximum of 
about 15-20 minutes for bms predictions, assuming the existence of significant 
points for that building in the 
configuration file, and may take longer for those without. 
Please use this information in your cloud scheduling planning.

 
# Contributors

* [David Karapetyan](mailto:david.karapetyan@gmail.com)

# License

* Proprietary
