# ml-it
Quickly find the right ML model and parameters for your structured data


I used Python 3.8.12, you can find library requierments in requirements.txt. 

Main folder contains following folders:

    data/
        contains input data
        

    eda/
        using ydata-profiling
        https://github.com/ydataai/ydata-profiling


    models/
        contains models trained in the notebook.


    utils/

        model_utils.py 
            A class, includes utility functions for fine-tuning hyperparameters and evaluating models

        data_utils.py
            A class, includes data utility functions doing EDA, projecting data to lower dimensions, normalization, and data generators. 

        config.py
            contains several pre-determined configurations for each model in order to do a search of hyperparameters.  


