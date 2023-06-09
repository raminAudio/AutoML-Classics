'''
Author: Ramin Anushiravani
Date: April 12th/23
Model Evaluation Helper
'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, plot_confusion_matrix, f1_score
import numpy as np
from sklearn import preprocessing
from utils.data_utils import DataUtil
from matplotlib.pylab import plt
from IPython.display import clear_output


class ModelUtil(DataUtil):
    '''
    This class contains methods for evaluating models and keeps track of hyperparameters. Inherits from the data utility class
    '''

    def __init__(self, verbose=0):
        self.hyperparameter_search = {}
        self.verbose = verbose

    def cross_validation(self, clf, X, y, num_folds=5):
        '''
        Perform cross validation on a Sklearn model
        Args:
            clf: sklearn model
            X: data
            y: labels
            num_folds : number of folds
            normaliza: normalizes data if set to 1, set to zero if data is already normalized
            valid: if set to 0 it will return the normalization model as a third argument to use offline
        '''

        valid_scores = []
        train_scores = []
        valid_probabitlies = []
        Gen = self.gen_data(X, y, num_folds)
        try:
            while True:
                training_data, training_label, validation_data, validation_label = next(
                    Gen)
                norm_scaler = preprocessing.StandardScaler().fit(training_data)
                training_data = norm_scaler.transform(training_data)
                validation_data = norm_scaler.transform(validation_data)

                clf.fit(training_data, training_label)  # Train model
                train_scores.append(
                    clf.score(training_data, training_label))  # training score
                valid_scores.append(
                    clf.score(
                        validation_data,
                        validation_label))  # validation sore
                valid_probabitlies.append(clf.predict_proba(
                    validation_data))  # validation probablities

        except StopIteration:
            done = 1

        if self.verbose:
            print(
                "training score",
                '%0.4F' %
                round(
                    np.max(train_scores),
                    3),
                "validation score",
                '%0.4F' %
                round(
                    np.mean(valid_scores),
                    3))

        return round(np.max(train_scores), 3), round(np.mean(valid_scores), 3)

    def find_best_hyperparameters(self, model_name):
        '''
        Looks through the hyperparameters used to train the model and returns the parameters that had the highest validation accuracy
        Args:
            model_name
        Returns:
            A list of dictionary, containing best hyperparameters from the config of this model
        '''
        if model_name != '':
            hyper_search = {}
            for k in self.hyperparameter_search.keys():
                for model in self.hyperparameter_search[k]:
                    if model['Model'] == model_name:
                        hyper_search[k] = model
        else:
            hyper_search = self.hyperparameter_search

        max_score = max(hyper_search.keys())
        use_param = hyper_search[max_score]
        if self.verbose:
            print(
                "Model {} \n {} \n validation score: {}".format(
                    model_name,
                    use_param,
                    '%0.4F' % max_score))

        return use_param

    def store_params(self, param, valid_score):
        '''
        Stores the parameters of the model with its corresponding validation score in a variable for book keeping
        '''
        if valid_score in self.hyperparameter_search.keys():
            self.hyperparameter_search[valid_score].append(param)
        else:
            self.hyperparameter_search[valid_score] = [param]

    def get_roc(self, predictions, ground_truth):
        '''
        Args:
            predictions  : probablities of class 1
            ground_truth : Test data labels
        Returns values needed for an ROC curve
        '''
        fpr, tpr, thr = roc_curve(ground_truth, predictions)
        auc = roc_auc_score(ground_truth, predictions)
        return fpr, tpr, auc, thr

    def plot_cm_roc(self, prediction_prob, ground_truth, name=''):
        '''
        plots Confusion matrix and ROC curve
        Args:
            prediction_prob  : probablities of class 1
            ground_truth : Test data labels
            name: title to use for fiugres
        '''
        _, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        if len(np.shape(prediction_prob)) > 1:
            t = prediction_prob[:, 1] > prediction_prob[:, 0]
            fpr, tpr, auc, thrs = self.get_roc(
                prediction_prob[:, 1], ground_truth)

        else:
            t = np.array(prediction_prob) > 0.5
            fpr, tpr, auc, thrs = self.get_roc(prediction_prob, ground_truth)

        cm = confusion_matrix(ground_truth, t)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[
                False, True]).plot(
            ax=axs[0])
        f1 = f1_score(ground_truth, t, average='weighted')
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1 - fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)

        plt.title(name +
                  ' F1 Score {} ROC - AUC {} and best Thr {}'.format('%0.4F' %
                                                                     f1, '%0.4F' %
                                                                     round(auc, 5), '%0.4F' %
                                                                     thrs[ix]))
        axs[1].plot(fpr, tpr)
        plt.show()

    def real_time_test(self, model, test_datas, test_labels, thr):
        '''
        helper function for visualizing model in real-time ish
        model with a predict method 0 or 1
        test_datas is normalized
        test_labels is ground truth labels
        '''
        idx = 0
        for data, label in zip(test_datas, test_labels):
            idx += 1
            pred = model.predict(data[None,], verbose=0)
            pred = pred.squeeze() > thr
            plt.figure(figsize=(15, 5), dpi=100)
            if label != pred.squeeze():
                c = 'r'
            else:
                c = 'b'
            plt.ylim([-10, 10])
            plt.plot(data, c)
            plt.title(
                "normalized data - sample {} label {} prediction {}".format(
                    idx, label, '%0.4F' % pred))
            plt.xlabel('sensors')
            plt.show()
            clear_output(wait=True)
