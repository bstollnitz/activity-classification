import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pywt
import torch
from torch.utils import data
import h5py
from tqdm import tqdm

import utils_graph
import utils_io
from signal_data import SignalData
from cnn import CNN
from feed_forward import FeedForward
from gram_data import GramData

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/activity-classification/'
S3_FILENAME = 'activity-dataset.zip'

DATA_FOLDER = 'data'
PLOTS_FOLDER = 'plots'

SPECTROGRAMS_IMAGES_FOLDER = 'spectrograms/images'
SPECTROGRAMS_DATA_FOLDER = 'spectrograms/data'
SPECTROGRAMS_TRAIN_FILE_NAME = 'train_spectrograms.hdf5'
SPECTROGRAMS_TEST_FILE_NAME = 'test_spectrograms.hdf5'

SCALEOGRAMS_IMAGES_FOLDER = 'scaleograms/images'
SCALEOGRAMS_DATA_FOLDER = 'scaleograms/data'
SCALEOGRAMS_TRAIN_FILE_NAME = 'train_scaleograms.hdf5'
SCALEOGRAMS_TEST_FILE_NAME = 'test_scaleograms.hdf5'


def ensure_reproducibility() -> None:
    """Ensures reproducibility of results.
    """
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)


def _calculate_accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates accuracy of multiclass prediction.
    """
    num_correct = (torch.argmax(output, dim=1) == labels).sum()
    num_train = len(labels)
    accuracy = (float(num_correct) / float(num_train)) * 100
    return accuracy


def scenario1(data: SignalData) -> None:
    """Uses a simple feed forward network to classify the raw signal.
    """
    print('Scenario 1: feed forward network on raw signal')

    input_size = data.num_timesteps * data.num_components
    feed_forward = FeedForward(input_size, input_size, data.num_activity_labels)
    print(feed_forward)

    train_signals = np.reshape(data.train_signals, (-1, input_size))
    train_signals = torch.from_numpy(train_signals).float()
    train_labels = torch.from_numpy(data.train_labels - 1).long()

    test_signals = np.reshape(data.test_signals, (-1, input_size))
    test_signals = torch.from_numpy(test_signals).float()
    test_labels = torch.from_numpy(data.test_labels - 1).long()

    optimizer = torch.optim.SGD(feed_forward.parameters(), lr=0.02)
    loss_function = torch.nn.NLLLoss()
    num_epochs = 3000

    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        # Zero-out the optimizer's gradients.
        optimizer.zero_grad()
        # Training data: foreward pass, backward pass, optimize.
        train_output = feed_forward(train_signals)
        train_loss = loss_function(train_output, train_labels)
        train_loss.backward()
        optimizer.step()
        # Print statistics.
        train_accuracy = _calculate_accuracy(train_output, train_labels)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss.item())
        print(f'Epoch: {epoch}. Loss: {train_loss:0.2f}. Accuracy: {train_accuracy:0.2f}%.')
        # Test data: predict.
        test_output = feed_forward(test_signals)
        test_loss = loss_function(test_output, test_labels)
        test_accuracy = _calculate_accuracy(test_output, test_labels)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss.item())

    utils_graph.graph_nn_results(train_accuracy_list, test_accuracy_list, 
        "Accuracy of prediction of signals", 
        "Accuracy", PLOTS_FOLDER, '1_accuracy.html')

    utils_graph.graph_nn_results(train_loss_list, test_loss_list, 
        "Loss of prediction of signals", 
        "Loss", PLOTS_FOLDER, '1_loss.html')


def _save_grams(signals: np.ndarray, file_name: str, gram_type: str):
    """Computes and saves spectrograms or scaleograms for all the signals.
    """
    if gram_type == 'spectrograms':
        data_folder = SPECTROGRAMS_DATA_FOLDER
        create_gram_func = _create_spectrogram
    elif gram_type == 'scaleograms':
        data_folder = SCALEOGRAMS_DATA_FOLDER
        create_gram_func = _create_scaleogram
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    gram_path = Path(data_folder, file_name)

    if not gram_path.exists():
        print(f'  Generating and saving {gram_type} to {file_name}.')
        Path(data_folder).mkdir(exist_ok=True, parents=True)
        # 2947 x 9 x 128
        (num_instances, num_components, num_timesteps) = signals.shape
        # 2947 x 9 x 128 x 128
        grams = np.zeros((num_instances, num_components,
            num_timesteps, num_timesteps))

        graph_gaussian_signal = True
        for instance in range(num_instances):
            for component in range(num_components):
                signal = signals[instance, component, :]
                # 128 x 128
                gram = create_gram_func(signal, graph_gaussian_signal)
                grams[instance, component, :, :] = gram
                graph_gaussian_signal = False

        with h5py.File(gram_path, 'w') as group:
            group.create_dataset(name=gram_type, shape=grams.shape, 
                dtype='f', data=grams)


def _save_gram_images(labels: np.ndarray, activity_names: dict,
    gram_type: str) -> None:
    """Saves a few spectrogram or scaleogram images for each component if this 
    hasn't been done already, for debugging purposes.
    Number of images saved: number of activities (6) x number of sets per
    activity (3) x number of components (9).
    """
    if gram_type == 'spectrograms':
        data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TEST_FILE_NAME)
        images_folder = Path(SPECTROGRAMS_IMAGES_FOLDER)
    elif gram_type == 'scaleograms':
        data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TEST_FILE_NAME)
        images_folder = Path(SCALEOGRAMS_IMAGES_FOLDER)
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')
    
    # Create images folder if it doesn't exist.
    images_folder.mkdir(exist_ok=True, parents=True)

    # Open data file.
    with h5py.File(data_path, 'r') as gram_file:
        # If there are no images in the folder:
        images = [item for item in images_folder.iterdir() if item.suffix == '.png']
        if len(images) == 0:
            print('  Saving images.')
            num_sets_per_activity = 3
            # Find all the unique activity numbers in our labels.
            activities = np.unique(labels)
            # For each activity present in the labels:
            for activity in activities:
                instance_indices = np.nonzero(labels == activity)[0][0:num_sets_per_activity]
                # For each instance of that activity:
                for instance_index in instance_indices:
                    # Read the image values from data file.
                    activity_grams = gram_file[gram_type][instance_index, :, :, :]
                    # For each of the 9 components: 
                    num_components = activity_grams.shape[0]
                    for component in range(num_components):
                        gram = activity_grams[component, :, :]
                        activity_name = activity_names[activity]
                        file_name = f'{activity_name}_{instance_index + 1}_{component + 1}.png'
                        # Save the spectrogram or scaleogram.
                        utils_io.save_image(gram, images_folder, file_name)


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray to values between 0 and 1.
    The max value maps to 1, but the min value may not hit 0.
    """
    return np.abs(my_array)/np.max(np.abs(my_array))


def get_trainval_generators(full_train_data_path: Path, train_labels: 
    np.ndarray, batch_size: int, num_workers: int) -> Tuple[data.DataLoader, 
    data.DataLoader]:
    """Splits the training images and labels into training and validation sets,
    and returns generators for those.
    """
    full_training_data = GramData(full_train_data_path, train_labels)
    full_training_len = len(full_training_data)
    training_len = int(full_training_len * 0.8)
    validation_len = full_training_len - training_len
    (training_data, validation_data) = data.random_split(full_training_data, 
        [training_len, validation_len])

    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers}
    training_generator = data.DataLoader(training_data, **params)
    validation_generator = data.DataLoader(validation_data, **params)

    return (training_generator, validation_generator)


def get_test_generator(test_data_path: Path, test_labels: 
    np.ndarray, batch_size: int, num_workers: int) -> data.DataLoader:
    """Gets generator for test data.
    """
    test_data = GramData(test_data_path, test_labels)
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_workers}
    test_generator = data.DataLoader(test_data, **params)
    return test_generator


def _classify_grams(full_train_labels: np.ndarray, test_labels: np.ndarray, 
    gram_type: str) -> None:
    """Classifies spectrograms or scaleograms using a CNN.
    """
    print('  Classifying images.')
    start_time = time.time()

    if gram_type == 'spectrograms':
        full_train_data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TRAIN_FILE_NAME)
        test_data_path = Path(SPECTROGRAMS_DATA_FOLDER, SPECTROGRAMS_TEST_FILE_NAME)
    elif gram_type == 'scaleograms':
        full_train_data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TRAIN_FILE_NAME)
        test_data_path = Path(SCALEOGRAMS_DATA_FOLDER, SCALEOGRAMS_TEST_FILE_NAME)
    else:
        raise Exception('gram_type must be "spectrograms" or "scaleograms"')

    # Crete CNN.
    cnn = CNN()
    print(f'  {cnn}')

    # There are 6 labels, and Pytorch expects them to go from 0 to 5.
    full_train_labels = full_train_labels - 1
    test_labels = test_labels - 1

    # Get training, validation and test generators.
    batch_size = 32
    num_workers = 0
    (training_generator, validation_generator) = get_trainval_generators(
        full_train_data_path, full_train_labels, batch_size, num_workers)
    test_generator = get_test_generator(test_data_path, test_labels, 
        batch_size, num_workers)

    # Cross entropy loss = softmax + negative log likelihood.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    max_epochs = 1

    # Use CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # With benchmark enabled, cuDNN figures out which algorithm is most 
    # performant for our model, and this speeds up performance for larger
    # CNNs. In this cenario it made no difference in performance, so we'll
    # leave the defaults.
    # torch.backends.cudnn.benchmark = True

    # Convert the model parameters to torch's float. 
    cnn = cnn.float().to(device)

    training_accuracy_list = []
    training_loss_list = []
    validation_accuracy_list = []
    validation_loss_list = []
    test_accuracy_list = []
    test_loss_list = []

    for epoch in range(max_epochs):
        print(f'Epoch {epoch}')

        # Training data.
        training_batch_num = 0
        training_cummulative_accuracy = 0
        training_cummulative_loss = 0
        for batch_data, batch_labels in tqdm(training_generator):
            # Transfer to GPU.
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.long().to(device)
            # Zero-out the optimizer's gradients.
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize.
            batch_output = cnn(batch_data.float())
            batch_loss = loss_function(batch_output, batch_labels)
            batch_loss.backward()
            optimizer.step()
            # Gather statistics.
            batch_accuracy = _calculate_accuracy(batch_output, batch_labels)
            training_cummulative_accuracy += batch_accuracy
            training_cummulative_loss += batch_loss.item()
            training_batch_num += 1

        training_avg_accuracy = training_cummulative_accuracy / training_batch_num
        training_avg_loss = training_cummulative_loss / training_batch_num
        training_accuracy_list.append(training_avg_accuracy)
        training_loss_list.append(training_avg_loss)
        print(f'  Training accuracy: {training_avg_accuracy:0.2f}' + 
            f'  Training loss: {training_avg_loss:0.2f}')

        # Validation data.
        validation_batch_num = 0
        validation_cummulative_accuracy = 0
        validation_cummulative_loss = 0
        for batch_data, batch_labels in tqdm(validation_generator):
            # Transfer to GPU.
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.long().to(device)
            # Predict.
            batch_output = cnn(batch_data.float())
            batch_loss = loss_function(batch_output, batch_labels)
            # Gather statistics.
            batch_accuracy = _calculate_accuracy(batch_output, batch_labels)
            validation_cummulative_accuracy += batch_accuracy
            validation_cummulative_loss += batch_loss.item()
            validation_batch_num += 1

        validation_avg_accuracy = validation_cummulative_accuracy / validation_batch_num
        validation_avg_loss = validation_cummulative_loss / validation_batch_num
        validation_accuracy_list.append(validation_avg_accuracy)
        validation_loss_list.append(validation_avg_loss)
        print(f'  Validation accuracy: {validation_avg_accuracy:0.2f}' + 
            f'  Validation loss: {validation_avg_loss:0.2f}')

    utils_graph.graph_nn_results(training_accuracy_list, validation_accuracy_list, 
        f'Accuracy of classification of {gram_type}', 
        'Accuracy', PLOTS_FOLDER, f'3_{gram_type}_accuracy.html')

    utils_graph.graph_nn_results(training_loss_list, validation_loss_list, 
        f'Loss of classification of {gram_type}', 
        'Loss', PLOTS_FOLDER, f'3_{gram_type}_loss.html')

    # Test data.
    for batch_data, batch_labels in tqdm(test_generator):
        # Transfer to GPU.
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.long().to(device)
        # Predict.
        batch_output = cnn(batch_data.float())
        batch_loss = loss_function(batch_output, batch_labels)
        # Gather statistics.
        batch_accuracy = _calculate_accuracy(batch_output, batch_labels)
        test_accuracy_list.append(batch_accuracy)
        test_loss_list.append(batch_loss.item())

    test_avg_accuracy = np.average(test_accuracy_list)
    test_avg_loss = np.average(test_loss_list)
    print(f'  Test accuracy: {test_avg_accuracy:0.2f}' + 
        f'  Test loss: {test_avg_loss:0.2f}')

    utils_io.print_elapsed_time(start_time, time.time())


def _get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """
    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def _graph_gaussian_signal(signal: np.ndarray, g: np.ndarray) -> None:
    """Saves a graph containing a signal and the Gaussian function used to 
    filter it.
    """
    # Plot Gaussian filter and signal overlayed in same graph.
    time_list = np.arange(len(signal))
    signal = _normalize(signal) 
    x = np.append([time_list], [time_list], axis=0)
    y = np.append([g], [signal], axis=0)
    utils_graph.graph_overlapping_lines(x, y, 
        ['Gaussian filter', 'Signal'],
        'Time', 'Amplitude', 
        'Example of a signal and corresponding Gaussian filter',
        PLOTS_FOLDER, '2_sample_gaussian_signal.html')


def _create_spectrogram(signal: np.ndarray, 
    graph_gaussian_signal: bool) -> np.ndarray:
    """Creates spectrogram for signal.
    """
    n = len(signal)
    # Times of the input signal.
    time_list = np.arange(n)
    # Horizontal axis of the output spectrogram (times where we will center the 
    # Gabor filter).
    time_slide = np.arange(n)
    # The vertical axis is the frequencies of the FFT, which is the same size
    # as the input signal.
    spectrogram = np.zeros((n, n), dtype=complex)
    for (i, time) in enumerate(time_slide):
        sigma = 3
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        ugt = np.fft.fftshift(np.fft.fft(ug))
        spectrogram[:, i] = ugt
        if i == n//2 and graph_gaussian_signal == True:
            _graph_gaussian_signal(signal, g)
    # We normalize to get real values between 0 and 1.
    spectrogram = _normalize(spectrogram)
    return spectrogram


def scenario2(data: SignalData) -> None:
    """Creates spectrograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 2: spectrograms + CNN')
    _save_grams(data.train_signals, SPECTROGRAMS_TRAIN_FILE_NAME, 'spectrograms')
    _save_grams(data.test_signals, SPECTROGRAMS_TEST_FILE_NAME, 'spectrograms')
    _save_gram_images(data.test_labels, data.activity_labels, 'spectrograms')

    _classify_grams(data.train_labels, data.test_labels, 'spectrograms')


def _create_scaleogram(signal: np.ndarray, graph_wavelet_signal: bool) -> np.ndarray:
    """Creates scaleogram for signal.
    """
    # Length of the signal: 128
    n = len(signal)
    time_list = np.arange(n)
    # Scale 1 corresponds to a wavelet of width 17 (lower_bound=-8, upper_bound=8).
    # Scale n corresponds to a wavelet of width n*17.
    scale_list = np.arange(0, n) / 8 + 1
    wavelet = 'mexh'
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    scaleogram = _normalize(scaleogram)

    if graph_wavelet_signal:
        signal = _normalize(signal)
        x = np.append([time_list], [time_list], axis=0)

        # Graph the narrowest wavelet together with the signal.
        [wav_narrowest, _] = pywt.ContinuousWavelet(wavelet).wavefun(
            length=n*int(scale_list[0])) 
        y = np.append([wav_narrowest], [signal], axis=0)
        utils_graph.graph_overlapping_lines(x, y, 
            ['Mexican hat wavelet', 'Signal'],
            'Time', 'Scale', 
            'Example of a signal and narrowest wavelet',
            PLOTS_FOLDER, '3_sample_narrowest_wavelet.html')

        # Graph the widest wavelet together with the signal.
        # wavefun gives us the original wavelet, with a width of 17 (scale=1).
        # We want to stretch that signal by scale_list[n-1].
        # So we oversample the wavelet computation and take the n points in 
        # the middle.
        [wav_widest, _] = pywt.ContinuousWavelet(wavelet).wavefun(
            length=n*int(scale_list[n-1]))
        middle = len(wav_widest) // 2
        lower_bound = middle - n // 2
        upper_bound = lower_bound + n 
        wav_widest = wav_widest[lower_bound:upper_bound]
        y = np.append([wav_widest], [signal], axis=0)
        utils_graph.graph_overlapping_lines(x, y, 
            ['Mexican hat wavelet', 'Signal'],
            'Time', 'Scale', 
            'Example of a signal and widest wavelet',
            PLOTS_FOLDER, '3_sample_widest_wavelet.html')

    return scaleogram

def _save_wavelets() -> None:
    """Saves three different kinds of mother wavelets to be used in the 
    theoretical part of the report.
    """
    n = 100
    wavelet_names = ['gaus1', 'mexh', 'morl']
    titles = ['Gaussian wavelet', 'Mexican hat wavelet', 'Morlet wavelet']
    file_names = ['gaussian.html', 'mexican_hat.html', 'morlet.html']
    for i in range(len(wavelet_names)):
        file_name = file_names[i]
        path = Path(PLOTS_FOLDER, file_name)
        if not path.exists():
            wavelet_name = wavelet_names[i]
            wavelet = pywt.ContinuousWavelet(wavelet_name)
            [wavelet_fun, x] = wavelet.wavefun(length=n)
            utils_graph.graph_2d_line(x, wavelet_fun, 
                'Time', 'Amplitude', titles[i], 
                PLOTS_FOLDER, file_name)


def scenario3(data: SignalData) -> None:
    """Creates scaleograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 3: scaleograms + CNN')
    _save_grams(data.train_signals, SCALEOGRAMS_TRAIN_FILE_NAME, 'scaleograms')
    _save_grams(data.test_signals, SCALEOGRAMS_TEST_FILE_NAME, 'scaleograms')
    _save_gram_images(data.test_labels, data.activity_labels, 'scaleograms')
    _save_wavelets()
    _classify_grams(data.train_labels, data.test_labels, 'scaleograms')


def main() -> None:
    """Main program.
    """
    ensure_reproducibility()
    data = SignalData(DATA_FOLDER, S3_URL, S3_FILENAME)
    # Scenario 1: 2-layer feed forward network on raw signal.
    # Epoch: 2999. Loss: 0.48. Accuracy: 79.92%.
    # scenario1(data)
    # Scenario 2: classify using CNN on spectrograms.
    # scenario2(data)
    # Scenario 3: classify using CNN on scaleograms.
    scenario3(data)


if __name__ == '__main__':
    main()
