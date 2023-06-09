'''
Author: Ramin Anushiravani
Date: April 11th/23
Data Uititly
'''

from IPython.display import clear_output
import warnings
import tensorflow
import logging
import pandas as pd
from matplotlib.pylab import plt
import numpy as np
from scipy.signal import find_peaks
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import pickle
import os
import librosa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tensorflow.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')


class DataUtil:
    '''
    This class contains method for loading, cleaning, and doing EDA on data
    '''

    def __init__(self, data=None, num_sensors=1024):
        self.sensor_i = data
        self.num_sensors = num_sensors

    def set_data(self, data):
        self.sensor_i = data

    def set_num_sensors(self, num_sensors):
        self.num_sensors = num_sensors

    def get_mean(self):
        return np.mean(self.sensor_i)

    def get_max(self):
        return np.max(self.sensor_i)

    def get_min(self):
        return np.min(self.sensor_i)

    def get_std(self):
        return np.std(self.sensor_i)

    def get_nan(self):
        return np.sum(np.isnan(self.sensor_i))

    def get_zero_count(self):
        return np.sum(self.sensor_i == 0)

    def get_zero_nzero(self):
        return self.get_zero_count() / len(self.sensor_i)

    def capture_sensor_stat(self, times_series_data):
        '''
        Takes in a time series and extract selected features from the signal
        Returns a dataframe
        '''
        sensor_stats = []
        for i in range(0, self.num_sensors):
            self.set_data(times_series_data[:, i])
            mean_sensor_i = self.get_mean()
            max_val_i = self.get_max()
            min_val_i = self.get_min()
            nan_count_i = self.get_nan()
            std_sensor_i = self.get_std()
            num_zeros_sensor_i = self.get_zero_nzero()
            zero_to_nonzero_senor_i = self.get_zero_nzero()
            stats_senor_i = {
                "sensor": i,
                "mean": mean_sensor_i,
                "std": std_sensor_i,
                "ztnz": zero_to_nonzero_senor_i,
                "max_val": max_val_i,
                "min_val": min_val_i,
                "nan_count": nan_count_i}
            sensor_stats.append(stats_senor_i)

        df_sensor_stat = pd.DataFrame(sensor_stats)

        return df_sensor_stat

    def plot_eda(self, sampled_data):
        '''
        plots EDA of a sampled dataset
        '''
        avg_of_all_samples = np.mean(sampled_data, axis=0)
        std_of_all_samples = np.std(sampled_data, axis=0)
        max_of_all_samples = np.max(sampled_data, axis=0)

        sample = sampled_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample)
        plt.title("Sample sensor data")

        norm_scaler = self.normalizer(sampled_data)
        sampled_norm_data = norm_scaler.transform(sampled_data)

        sample_norm = sampled_norm_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_norm)
        plt.title("Sample normalized sensor data")

        pca_model = self.project_pca(sampled_norm_data, dim=64)
        pca_data = pca_model.transform(sampled_norm_data)
        sample_pca = pca_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_pca)
        plt.title("Sample 128 PCA-ed sensor data")

        eng_data = self.feature_engr(sampled_norm_data)
        sample_eng = eng_data[4, :]
        plt.figure(figsize=(15, 5), dpi=100)
        plt.plot(sample_eng)
        plt.title("Sample ENG-ed sensor data")

        plt.figure(figsize=(15, 5), dpi=100)
        plt.imshow(sampled_data, origin='lower', cmap='gray_r')
        plt.plot(avg_of_all_samples + 5, 'r')
        plt.plot(std_of_all_samples + 15, 'g')
        plt.plot(max_of_all_samples + 25, 'k')
        plt.legend(['mean', 'std', 'max'])
        plt.xlabel("sensors")
        plt.title("Sampled Data Stats")

        plt.figure(figsize=(15, 5), dpi=100)
        peaks_mu, _ = find_peaks(avg_of_all_samples, height=5)
        plt.plot(avg_of_all_samples)
        plt.plot(peaks_mu, avg_of_all_samples[peaks_mu], "x")
        plt.plot(std_of_all_samples)
        print("peak mu >5 {}".format(peaks_mu))
        plt.title("Sampled Data Peaks")
        plt.xlabel("sensors")
        plt.show()

    def stat_window(self, data, jump=128):
        '''
        window over sensors i.e., average them out to identify if there are any patterns of interest
        75% sensor overlap
        '''
        b = [0, jump]
        mean_chunk_i = []
        std_chunk_i = []
        while b[1] <= len(data):
            chunck_i = data[b[0]:b[1]]
            self.set_data(chunck_i)
            mean_chunk_i.append(self.get_mean())
            std_chunk_i.append(self.get_std())
            b[0] += int(jump / 4)
            b[1] += int(jump / 4)

        return mean_chunk_i, std_chunk_i

    def make_sparse(self, data):
        '''
        converts data to sklearn sparse
        '''
        return scipy.sparse.csr_matrix(data)

    def feature_engr(self, ys):
        '''
        Feature engineering
        '''
        feats = []
        if len(np.shape(ys)) < 1:
            ys = [ys]
        for idx in range(np.shape(ys)[0]):
            y = ys[idx, :]
            zx = librosa.feature.zero_crossing_rate(
                y, frame_length=128, hop_length=64)[0]
            rms = librosa.feature.rms(y=y, frame_length=128, hop_length=64)[0]
            spec_flat = librosa.feature.spectral_flatness(
                y=y, n_fft=128, hop_length=64)[0]
            zxF = librosa.feature.zero_crossing_rate(y)[0]
            rmsF = librosa.feature.rms(y=y)[0]
            spec_flatF = librosa.feature.spectral_flatness(y=y)[0]
            self.set_data(y)
            feature = np.concatenate(
                (zx, rms, spec_flat, zxF, rmsF, spec_flatF))
            feats.append(np.tanh(feature))
        return np.array(feats)

    def project_tsne(self, data, dim=3):
        '''
        project data to 3 dim using T-SNE
        '''
        data_transformation = TSNE(
            n_components=dim,
            init='random',
            perplexity=30,
            n_iter=400,
            learning_rate=5).fit_transform(data)
        return data_transformation

    def project_umap(self, data, dim=3):
        '''
        project data to 3 dim using UMAP
        '''
        mapper = umap.UMAP(n_components=dim, init='random').fit(data)
        data_transformation = mapper.transform(data)
        return data_transformation

    def project_pca(self, data, dim=3):
        '''
        project data to 3 dim using PCA
        '''
        pca = PCA(n_components=dim)
        return pca.fit(data)
    
    def project_nmf(self,data,dim=128):
        model = NMF(n_components=dim, init='random', random_state=0)
        return model.fit_transform(data)

    def scatter_plot(self, data_transformation, labels):
        '''
        Args:
            data_transformation: a 3 dim matrix
            labels : associated labels to each sample in data_transformation
        Help for scatter plot of lower dim data
        '''
        if data_transformation.shape[1] == 3:
            x = data_transformation[:, 0]
            y = data_transformation[:, 1]
            z = data_transformation[:, 2]

            x0 = np.min(x) - np.median(x)
            x1 = np.max(x) - np.median(x)
            y0 = np.min(y) - np.median(y)
            y1 = np.max(y) - np.median(y)
            z0 = np.min(z) - np.median(z)
            z1 = np.max(z) - np.median(z)
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               marker=dict(
                                                   size=2,
                                                   color=labels,
                                                   colorscale='Viridis',
                                               ),
                                               text=labels,
                                               name='test')])
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        nticks=4, range=[
                            x0, x1],), yaxis=dict(
                        nticks=4, range=[
                            y0, y1],), zaxis=dict(
                        nticks=4, range=[
                            z0, z1],),), width=700, margin=dict(
                    r=20, l=10, b=10, t=10))
            fig.show()
        elif data_transformation.shape[1] == 2:
            x = data_transformation[:, 0]
            y = data_transformation[:, 1]
            fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='markers',
                                             marker=dict(
                                                 size=2,
                                                 color=labels,
                                                 colorscale='Viridis',
                                             ),
                                             text=labels,
                                             name='test')])
            fig.show()

    def extract_stft(self, data, n_fft=32):
        '''
        extract short time fourier transform from a time series
        returns magnitude and phase
        '''
        S = librosa.stft(data, n_fft=n_fft, hop_length=n_fft // 2)
        magnitude, phase = librosa.magphase(S)
        return magnitude, phase

    def extract_psd(self, magnitude, phase):
        '''
        extract and plot PSD from STFT
        return PSD
        '''
        phase_angle = np.angle(phase)
        librosa.display.specshow(magnitude, cmap='gray_r')
        avg_magnitude = np.mean(magnitude, axis=0)
        return avg_magnitude

    def load_csv(self, filename):
        '''
        load a csv file and fills missing values using padding from previous values
        '''
        df = pd.read_csv(filename)
        df_filling_missing_values = df.fillna(method="pad")
        return df_filling_missing_values

    def normalizer(self, data):
        '''
        takes in data and returns a sklearn normalizer to be used to normalize data
        '''
        norm_scaler = preprocessing.StandardScaler().fit(data)

        return norm_scaler

    def load_norm_data(self, filename):
        '''
        loads a csv file, assign 80% of data to training and 20% to testing
        normalizes training set and uses that normalization on testing set
        returns normalizes training and test data along with their corresponding labels
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        df_training = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                        0, 1, 2, 3])]
        df_testing = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                       4])]

        x_train = df_training[x_cols].to_numpy()
        y_train = df_training['y'].to_numpy()

        x_test = df_testing[x_cols].to_numpy()
        y_test = df_testing['y'].to_numpy()

        norm_scaler = self.normalizer(x_train)
        x_train_norm = norm_scaler.transform(x_train)
        x_test_norm = norm_scaler.transform(x_test)

        return x_train_norm, y_train, x_test_norm, y_test
    
    def load_data(self, filename):
        '''
        loads a csv file, assign 80% of data to training and 20% to testing
        normalizes training set and uses that normalization on testing set
        returns normalizes training and test data along with their corresponding labels
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        df_training = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                        0, 1, 2, 3])]
        df_testing = df_filling_missing_values[df_filling_missing_values['block'].isin([
                                                                                       4])]

        x_train = df_training[x_cols].to_numpy()
        y_train = df_training['y'].to_numpy()

        x_test = df_testing[x_cols].to_numpy()
        y_test = df_testing['y'].to_numpy()

        return x_train, y_train, x_test, y_test

    def gen_data(self, X, y, num_folds=5):
        '''
        Since data is a time-ordered list of samples, data is not randomized in time order. Validation set is extracted to maintain the time-order
        generates training and validation data data for cross-validation
        Returns a generator functions which returns training and validation data along with their labels

        '''
        assert len(
            y) % num_folds == 0, "number of samples must be divisible by num_folds"
        assert num_folds > 1, "number of folds must be bigger than 1"

        num_samples = int(len(y) / num_folds)
        b = [0, num_samples]
        while b[1] <= len(y):
            validation_data = X[b[0]:b[1],]
            validation_label = y[b[0]:b[1]]
            training_data = np.concatenate((X[0:b[0], :], X[b[1]:,]))
            training_label = np.concatenate((y[0:b[0]], y[b[1]:]))
            b[0] += num_samples
            b[1] += num_samples
            yield training_data, training_label, validation_data, validation_label

    def input_data(self, X_train, feature_name, X_test=[], num_feat=128):
        '''
        X_train is training data - this method does normalize data, make sure to send in normalized data if 'pca' is passed
        Takes in training data (and testing data if available) and returns them as is if feature_name is set to 'raw'
        and #num_feat 'pca' features if set to 'pca'
        '''
        if feature_name == 'pca':
            pca = self.project_pca(X_train, dim=num_feat)
            Xf_train = pca.transform(X_train)  # eigen feature
            if len(X_test):
                Xf_test = pca.transform(X_test)
            else:
                Xf_test = X_test

            return Xf_train, Xf_test

        elif feature_name == 'eng':
            feat_train = self.feature_engr(X_train)
            feat_test = self.feature_engr(X_test)
            return feat_train, feat_test
        elif feature_name == 'nmf':
            feat_train = self.project_nmf(X_train)
            feat_test = self.feature_engr(X_test)
            return feat_train, feat_test
            
        else:
            return X_train, X_test

    def viz_frames(self, X, y, num_frames=32):
        '''
        helper function for visualizing data as a series of frames
        takes in data and labels along with how many frames
        '''
        b = [0, num_frames]
        while b[1] <= len(X):
            fig, axs = plt.subplots(2, 1, figsize=(5, 10))
            axs[0].plot(np.mean(X[b[0]:b[1], :],axis=1))
            axs[0].set_title('50% overlap')
            axs[0].set_xlabel('Time Frames')
            axs[0].set_ylabel('Avg over sensor')
            axs[1].plot(y[b[0]:b[1]])
            axs[1].set_xlabel('Time Frames')
            axs[1].set_ylabel('Features')
            axs[1].set_title('Label')
            axs[1].set_ylim([-0.1, 1.5])
            plt.show()
            b[0] += int(num_frames / 2)
            b[1] += int(num_frames / 2)

            clear_output(wait=True)

    def viz_raw(self, X, y):
        '''
        helper function for visualizing data as samples
        takes in data and labels
        '''
        for idx, x in enumerate(X):
            label = y[idx]
            plt.figure(figsize=(25, 5), dpi=100)

            if label == 1:
                plt.plot(self.stat_window(x)[0], 'r')
                plt.plot(self.stat_window(x)[1], 'g')
            else:
                plt.plot(self.stat_window(x)[0])
                plt.plot(self.stat_window(x)[1])

            peaks_idx, _ = find_peaks(x, height=3)

            plt.title(
                "frame {} peak sensor {} label {}".format(
                    idx, peaks_idx, label))
            plt.ylim([-1, 5])
            plt.legend(['mean', 'std'])
            plt.xlabel('sensors')
            plt.show()
            clear_output(wait=True)

    def viz_projected_data(self, filename, projection='pca', mask=[True]*1024):
        '''
        helper function for visualizing low-dim projected data in 3D
        takes in a csv filename
        '''
        df_filling_missing_values = self.load_csv(filename)
        x_cols = [col for col in df_filling_missing_values.columns if 'x' in col]
        zero_data = df_filling_missing_values[df_filling_missing_values['y'] == 0]
        one_data = df_filling_missing_values[df_filling_missing_values['y'] == 1]

        times_series_data_0 = np.array([x[mask] for x in list(zero_data[x_cols].to_numpy())])
        times_series_data_1 = np.array([x[mask] for x in list(one_data[x_cols].to_numpy())])
        
        print(times_series_data_0.shape)
        data = np.concatenate((times_series_data_0, times_series_data_1))

        labels0 = np.zeros((1, len(times_series_data_0)))[0]
        labels1 = np.zeros((1, len(times_series_data_1)))[0] + 1
        labels = np.concatenate((labels0, labels1))

        c = list(zip(data, labels))
        np.random.shuffle(c)
        data, labels = zip(*c)
        data = np.array(data)

        norm_scaler = self.normalizer(data)
        data_norm = norm_scaler.transform(data)
        if projection == 'pca':
            projection = self.project_pca(data_norm, dim=3)
            data_transformation = projection.transform(data_norm)
            self.scatter_plot(data_transformation, labels)
        elif projection == 'umap':
            data_transformation = self.project_umap(data_norm, dim=3)
            self.scatter_plot(projected, labels)
        elif projection == 'tsne':
            data_transformation = self.project_tsne(data_norm, dim=3)
            self.scatter_plot(data_transformation, labels)
        elif projection == 'nmf':
            data_transformation = self.project_nmf(data, dim=3)
            self.scatter_plot(data_transformation, labels)

        return data_transformation
