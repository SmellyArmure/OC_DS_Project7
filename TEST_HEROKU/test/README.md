# First attempt to deploy my flask api to Heroku

##############################################
## Running Locally

```shell

# clone my git repo
$ git clone https://github.com/SmellyArmure/PROJECT7/XXXXXXX/WWWWWWWWWW.git

# se placer dans le répertoire du fichier d'où il faudra lancer l'API
$ cd WEB

# create the test_venv virtual environment using venv (built-in python)
# (on Windows) :
$ python -m venv test_venv

# activate the virtual environment (Windows)
C:\> <venv>\Scripts\activate.bat

# install requirements
$ pip install -r requirements.txt


# run the app
$ python api_flask.py migrate
$ python api_flask.py collectstatic

$ heroku local
```

Your app should now be running on [localhost:5000](http://localhost:5000/).

##############################################
## Deploying to Heroku

```shell
$ heroku create
$ git push heroku main

$ heroku run python manage.py migrate
$ heroku open
```
