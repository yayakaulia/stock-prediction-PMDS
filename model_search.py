{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn import model_selection\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.svm import SVR\n",
    "from sklearn import metrics\n",
    "import random\n",
    "import yaml\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "f = open(\"params.yaml\", \"r\")\n",
    "params = yaml.load(f, Loader=yaml.SafeLoader)\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_data(params):\n",
    "    x_train = joblib.load(params['DUMP_TRAIN'])\n",
    "    y_train = joblib.load(params['Y_PATH_TRAIN'])\n",
    "    x_valid = joblib.load(params['DUMP_VALID'])\n",
    "    y_valid = joblib.load(params['Y_PATH_VALID'])\n",
    "\n",
    "    return x_train, y_train, x_valid, y_valid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import RandomizedSearchCV\n",
    "from sklearn import model_selection\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso\n",
    "from sklearn.svm import LinearSVR\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection import TimeSeriesSplit\n",
    "import yaml\n",
    "\n",
    "x_train, y_train, x_valid, y_valid  = read_data(params)\n",
    "\n",
    "def model_search(x_train, y_train):\n",
    "    ts_cv = TimeSeriesSplit(\n",
    "        n_splits=5,\n",
    "        max_train_size=None,\n",
    "    )\n",
    "\n",
    "    models = []\n",
    "    models.append(('RandomForrest', RandomForestRegressor()))\n",
    "    models.append(('Lasso', Lasso()))\n",
    "    models.append(('Ridge', Ridge()))\n",
    "    models.append(('SVR', LinearSVR()))\n",
    "    \n",
    "    results = []\n",
    "    names = []\n",
    "    scoring = params[\"scoring\"]\n",
    "    for name, model in models:\n",
    "        #kfold = model_selection.KFold(n_splits=5, shuffle=False)\n",
    "        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=ts_cv, scoring=scoring)\n",
    "        results.append(cv_results)\n",
    "        names.append(name)\n",
    "    results_mean = [sum(x)/5 for x in results]\n",
    "    df = pd.DataFrame(list(zip(names, results_mean)), columns =['model', 'score'])\n",
    "    df = df.set_index('model')\n",
    "    max_model = df.idxmax()\n",
    "    best_model = max_model[0]\n",
    "    print(best_model)\n",
    "    joblib.dump(best_model, 'output/model_name.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ridge\n"
     ]
    }
   ],
   "source": [
    "yut = model_search(x_train, y_train)\n",
    "yut"
   ]
  }
 ],
 "metadata": {
  "interpreter": {
   "hash": "49de8e361bcaa7b87b0d9a1948e17b94b1b6765468a847b75af4a4273d6c7723"
  },
  "kernelspec": {
   "display_name": "Python 3.8.10 64-bit (windows store)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
