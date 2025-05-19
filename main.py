from utilities import *
import fit_model
import forecast
import test

experiments = ['1_2_6','2_4_5','3_7_0']

for experiment in experiments:
    fit_model.main(experiment)
    forecast.main(experiment)
    test.main(experiment)