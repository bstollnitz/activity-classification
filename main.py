from typing import Dict, List, Tuple
import random
from pathlib import Path

import numpy as np
import torch

from activity_data import ActivityData
from feed_forward import FeedForward
import utils_graph
import utils_io

S3_URL = 'https://bea-portfolio.s3-us-west-2.amazonaws.com/activity-classification/'
S3_FILENAME = 'activity-dataset.zip'

DATA_FOLDER = 'data'
SPECTROGRAMS_IMAGES_FOLDER = 'spectrograms/images'
SPECTROGRAMS_DATA_FOLDER = 'spectrograms/data'
PLOTS_FOLDER = 'plots'


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


def scenario1(data: ActivityData) -> None:
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
        optimizer.zero_grad()
        train_output = feed_forward(train_signals)
        train_loss = loss_function(train_output, train_labels)
        train_loss.backward()
        optimizer.step()
        train_accuracy = _calculate_accuracy(train_output, train_labels)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_loss.item())
        print(f'Epoch: {epoch}. Loss: {train_loss:0.2f}. Accuracy: {train_accuracy:0.2f}%.')

        test_output = feed_forward(test_signals)
        test_loss = loss_function(test_output, test_labels)
        test_accuracy = _calculate_accuracy(test_output, test_labels)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss.item())

    utils_graph.plot_nn_results(train_accuracy_list, test_accuracy_list, 
        "Accuracy of prediction of signals", 
        "Accuracy", PLOTS_FOLDER, '1_accuracy.html')

    utils_graph.plot_nn_results(train_loss_list, test_loss_list, 
        "Loss of prediction of signals", 
        "Loss", PLOTS_FOLDER, '1_loss.html')


def _get_gaussian_filter(b: float, b_list: np.ndarray, 
    sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at b, with standard 
    deviation sigma.
    """
    a = 1/(2*sigma**2)
    return np.exp(-a*(b_list-b)**2)


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray to real values between 0 and 1.
    """
    return np.abs(my_array)/np.max(np.abs(my_array))


def _create_spectrogram(signal: np.ndarray) -> np.ndarray:
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
    spectrogram = np.zeros((n, n))
    for (i, time) in enumerate(time_slide):
        sigma = 0.01
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        ugt = np.fft.fftshift(np.fft.fft(ug))
        spectrogram[:, i] = _normalize(ugt)

    return spectrogram


def _get_spectrograms(signals: np.ndarray, file_name: str) -> np.ndarray:
    """Loads or computes spectrograms for all the signals.
    """
    spectrogram_path = Path(SPECTROGRAMS_DATA_FOLDER, file_name)

    if not spectrogram_path.exists():
        print(f'  Generating and saving spectrograms to {file_name}.')
        # len x 128 x 9
        (num_instances, num_timesteps, num_components) = signals.shape
        # len x 128 x 128 x 9
        spectrograms = np.zeros((num_instances, num_timesteps, num_timesteps, 
            num_components))

        for instance in range(num_instances):
            for component in range(num_components):
                signal = signals[instance, :, component]
                # 128 x 128
                spectrogram = _create_spectrogram(signal)
                spectrograms[instance, :, :, component] = spectrogram

        Path(SPECTROGRAMS_DATA_FOLDER).mkdir(exist_ok=True)
        np.save(spectrogram_path, spectrograms)

    else:
        print(f'  Reading spectrograms from {file_name}.')
        spectrograms = np.load(spectrogram_path)

    return spectrograms


def _save_spectrogram_images(spectrograms: np.ndarray, labels: np.ndarray,
    activity_names: dict) -> None:
    """Saves a few spectrogram images for each component if this hasn't been 
    done already.
    """
    spectrograms_path = Path(SPECTROGRAMS_IMAGES_FOLDER)
    spectrograms_path.mkdir(exist_ok=True)

    # If there are no images in the spectrograms folder:
    images = [item for item in spectrograms_path.iterdir() if item.suffix == '.png']
    if len(images) == 0:
        print('  Saving spectrogram images.')
        # Find an instance of each activity.
        activities = np.unique(labels)
        for activity in activities:
            instance_index = np.nonzero(labels == activity)[0][0]
            activity_spectrograms = spectrograms[instance_index, :, :, :]
            # Save the 9 component spectrograms for that activity.
            num_components = activity_spectrograms.shape[2]
            for component in range(num_components):
                spectrogram = activity_spectrograms[:, :, component]
                activity_name = activity_names[activity]
                file_name = f'2_{activity_name}_{component + 1}.png'
                utils_io.save_image(spectrogram, SPECTROGRAMS_IMAGES_FOLDER, file_name)


def _classify_images(train_images: np.ndarray, train_labels: np.ndarray,
    test_images: np.ndarray, test_labels: np.ndarray) -> None:
    pass


def scenario2(data: ActivityData) -> None:
    """Creates spectrograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 2: spectrograms + CNN')
    print('  Training data')
    train_spectrograms = _get_spectrograms(data.train_signals, 
        '2_train_spectrograms.npy')
    print('  Test data')
    test_spectrograms = _get_spectrograms(data.test_signals,
        '2_test_spectrograms.npy')
    _save_spectrogram_images(test_spectrograms, data.test_labels, 
        data.activity_labels)
    # _classify_images(train_spectrograms, data.train_labels, test_spectrograms, 
    #     data.test_labels)


def _compute_scaleograms(data: np.ndarray) -> np.ndarray:
    return data

def scenario3(data: ActivityData) -> None:
    """Creates scaleograms for each of the signals, and uses a CNN to 
    classify them.
    """
    print('Scenario 3: scaleograms + CNN')
    train_scaleograms = _compute_scaleograms(data.train_signals)
    test_scaleograms = _compute_scaleograms(data.test_signals)
    _classify_images(train_scaleograms, data.train_labels, test_scaleograms, 
        data.test_labels)


def main() -> None:
    """Main program.
    """
    ensure_reproducibility()
    data = ActivityData(DATA_FOLDER, S3_URL, S3_FILENAME)
    # Scenario 1: 2-layer feed forward network on raw signal.
    # Epoch: 2999. Loss: 0.48. Accuracy: 79.92%.
    # scenario1(data)
    # Scenario 2: classify using CNN on spectrograms.
    scenario2(data)
    # Scenario 3: classify using CNN on scaleograms.
    scenario3(data)


if __name__ == '__main__':
    main()
