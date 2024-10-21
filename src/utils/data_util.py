import tensorflow as tf
from sklearn import model_selection
from tensorflow.keras import backend as K  # noqa: N812
import json
from tqdm import tqdm, trange
import os
from numpy import random
import numpy as np
import shutil
import h5py
from pathlib import Path
from src.dnn_testing.eAI.utils.eai_dataset import EAIDataset

def test(model, x, y, verbose=True, batch_size=32):
    # Obtain accuracy as evaluation result of DNN model with test dataset
    score = model.evaluate(x, y, verbose=verbose, batch_size=batch_size)
    return score
def load_model_from_tf(model_dir: Path):
    model = tf.keras.models.load_model(model_dir)
    return model
def _parse_test_results(test_images, test_labels, results):
    successes = {}
    failures = {}
    dataset_len = len(test_labels)

    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()
        if predicted_label != test_label_index:
            if test_label_index not in failures:
                failures[test_label_index] = {}
            if predicted_label not in failures[test_label_index]:
                failures[test_label_index][predicted_label] = []
            failures[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )
        else:
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label, "index": i})
    return successes, failures


def _parse_test_results_both(test_images, test_labels, results, unfav=0, decision_threshold = 0.5, decision_boundary = 0.05):
    successes = {}
    failures = {}
    failures_id = []
    successes_id = []
    failures_close = {}
    failures_middle = {}
    failures_close_id = []
    #failures_middle2 = {}
    failures_far = {}
    failures_far_id = []
    successes_close = {}
    successes_middle = {}
    successes_close_id = []
    #successes_middle2 = {}
    successes_far = {}
    successes_far_id = []
    data_close = {}
    data_close_id = []
    data_middle = {}
    #data_middle2 = {}
    data_far = {}
    data_far_id = []
    data_all = {}
    data_id = []

    dataset_len = len(test_labels)

    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()
        data_id.append(i)
        if (results[i][0] <= (decision_threshold + decision_boundary) and predicted_label==0) or (results[i][1] <= (
                decision_threshold + decision_boundary) and predicted_label==1):
            if test_label_index not in data_close:
                data_close[test_label_index] = {}
            if predicted_label not in data_close[test_label_index]:
                data_close[test_label_index][predicted_label] = []
            data_close[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )
            data_close_id.append(i)
        else:
            if test_label_index not in data_middle:
                data_middle[test_label_index] = {}
            if predicted_label not in data_middle[test_label_index]:
                data_middle[test_label_index][predicted_label] = []
            data_middle[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )

        if (results[i][0] >= 0.8 and predicted_label == 0) or (results[i][1] >= 0.8 and predicted_label == 1):
            data_far_id.append(i)
            if test_label_index not in data_far:
                data_far[test_label_index] = {}
            if predicted_label not in data_far[test_label_index]:
                data_far[test_label_index][predicted_label] = []
            data_far[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )

        if predicted_label != test_label_index:
            failures_id.append(i)
            if (results[i][0] <= (decision_threshold + decision_boundary) and predicted_label == 0) or (
                    results[i][1] <= (
                    decision_threshold + decision_boundary) and predicted_label == 1):
                failures_close_id.append(i)
                if test_label_index not in failures_close:
                    failures_close[test_label_index] = {}
                if predicted_label not in failures_close[test_label_index]:
                    failures_close[test_label_index][predicted_label] = []
                failures_close[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
            else:
                if test_label_index not in failures_middle:
                    failures_middle[test_label_index] = {}
                if predicted_label not in failures_middle[test_label_index]:
                    failures_middle[test_label_index][predicted_label] = []
                failures_middle[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
            if (results[i][0] >= 0.8 and predicted_label == 0) or (results[i][1] >= 0.8 and predicted_label == 1):
                failures_far_id.append(i)

                if test_label_index not in failures_far:
                    failures_far[test_label_index] = {}
                if predicted_label not in failures_far[test_label_index]:
                    failures_far[test_label_index][predicted_label] = []
                failures_far[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
            if test_label_index not in failures:
                failures[test_label_index] = {}
            if predicted_label not in failures[test_label_index]:
                failures[test_label_index][predicted_label] = []
            failures[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )
            ## todo: saves all the valid datacin the dict
            if test_label_index not in data_all:
                data_all[test_label_index] = {}
            if predicted_label not in data_all[test_label_index]:
                data_all[test_label_index][predicted_label] = []
            data_all[test_label_index][predicted_label].append({"image": test_image, "label": test_label, "index": i})
        else:
            successes_id.append(i)
            if (results[i][0] <= (decision_threshold + decision_boundary) and predicted_label == 0) or (
                    results[i][1] <= (
                    decision_threshold + decision_boundary) and predicted_label == 1):
                successes_close_id.append(i)
                if test_label_index not in successes_close:
                    successes_close[test_label_index] = []
                successes_close[test_label_index].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
            else:
                if test_label_index not in successes_middle:
                    successes_middle[test_label_index] = []
                successes_middle[test_label_index].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
            if (results[i][0] >= 0.8 and predicted_label == 0) or (results[i][1] >= 0.8 and predicted_label == 1):
                if test_label_index not in successes_far:
                    successes_far[test_label_index] = []
                successes_far[test_label_index].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
                successes_far_id.append(i)
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label, "index": i})
            ## todo: saves all the valid datacin the dict
            if test_label_index not in data_all:
                data_all[test_label_index] = {}
            if predicted_label not in data_all[test_label_index]:
                data_all[test_label_index][predicted_label] = []
            data_all[test_label_index][predicted_label].append({"image": test_image, "label": test_label, "index": i})
    ## Get the mid datapoints
    failures_close_id.extend(failures_far_id)
    failed_mid2_id = [i for i in failures_id if i not in failures_close_id]
    failures_middle2 = {}
    for i in failed_mid2_id:
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()
        result = results[i]
        predicted_label = result.argmax()

        if test_label_index not in failures_middle2:
            failures_middle2[test_label_index] = {}
        if predicted_label not in failures_middle2[test_label_index]:
            failures_middle2[test_label_index][predicted_label] = []
        failures_middle2[test_label_index][predicted_label].append(
            {"image": test_image, "label": test_label, "index": i}
        )

    successes_close_id.extend(successes_far_id)
    successes_mid2_id = [i for i in successes_id if i not in successes_close_id]

    data_close_id.extend(data_far_id)
    data_mid2_id = [i for i in data_id if i not in data_close_id]

    all_data_id = [i for i in data_mid2_id]
    all_data_id.extend(failed_mid2_id)
    all_data_id.extend(successes_mid2_id)
    failures_middle2 = {}
    successes_middle2 = {}
    data_middle2 = {}
    for i in np.unique(all_data_id):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()
        result = results[i]
        predicted_label = result.argmax()

        if i in failed_mid2_id:
            if test_label_index not in failures_middle2:
                failures_middle2[test_label_index] = {}
            if predicted_label not in failures_middle2[test_label_index]:
                failures_middle2[test_label_index][predicted_label] = []
            failures_middle2[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )
        if i in successes_mid2_id:
            if test_label_index not in successes_middle2:
                successes_middle2[test_label_index] = []
            successes_middle2[test_label_index].append(
                {"image": test_image, "label": test_label, "index": i}
            )
        if i in data_mid2_id:
            if test_label_index not in data_middle2:
                data_middle2[test_label_index] = {}
            if predicted_label not in data_middle2[test_label_index]:
                data_middle2[test_label_index][predicted_label] = []
            data_middle2[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i}
            )

    return successes, failures, data_all, successes_close, failures_close, data_close, successes_middle, failures_middle, data_middle, successes_middle2, failures_middle2, data_middle2, successes_far, failures_far, data_far

def _sample_sensitive(input_pos, test_labels, pred_labels):
    #unfav, W_M, W_F, W_MC, W_MM, W_FC, W_FM, MC, MM, FC, FM = _get_unfav(
    #    np.argmax(test_labels, axis=1), np.argmax(pred_labels, axis=1))
    true_labels = np.argmax(input_pos[1], axis=1)
    unique_val, unique_counts = np.unique(true_labels, return_counts=True)

    neg_chosen_list = None
    for unique, val in zip(unique_val, unique_counts):
        indices_i = np.where(true_labels == unique)[0]

        W_FC, W_FM = _get_bias_weight_for(test_labels, pred_labels, attrib=unique)
        if val == np.max(unique_counts):
            id_ = np.random.choice(indices_i, size=int(W_FC*val))
            if neg_chosen_list == None:
                neg_chosen_list = (input_pos[0][id_], input_pos[1][id_], np.argmax(input_pos[1][id_], axis=1).reshape(-1,1))
            else:
                neg_chosen_list = (
                np.vstack((input_pos[0][id_], neg_chosen_list[0])), np.vstack((input_pos[1][id_], neg_chosen_list[1])),
                np.vstack((np.argmax(input_pos[1][id_], axis=1).reshape(-1,1), neg_chosen_list[2])))
        else:
            if neg_chosen_list == None:
                neg_chosen_list = (
                input_pos[0][indices_i], input_pos[1][indices_i], np.argmax(input_pos[1][indices_i], axis=1).reshape(-1, 1))
            else:
                neg_chosen_list = (
                    np.vstack((input_pos[0][indices_i], neg_chosen_list[0])),
                    np.vstack((input_pos[1][indices_i], neg_chosen_list[1])),
                    np.vstack((np.argmax(input_pos[1][indices_i], axis=1).reshape(-1, 1), neg_chosen_list[2])))

    return neg_chosen_list
def sample_negative(input_neg_):
    #print('labels: ', input_neg_[1])
    #print(input_neg_[1].shape)
    #if
    neg_labels = np.argmax(input_neg_[1], axis=1)
    unique_val, unique_counts = np.unique(neg_labels, return_counts=True)
    min_val = np.min(unique_counts)
    max_val = np.min(unique_counts)
    neg_chosen_list = None
    unfav = unique_val[0]
    i = 0
    for unique in unique_val:
        if unique_counts[i] == np.max(unique_counts):
            unfav = unique_val[i]
        indices_i = np.where(neg_labels == unique)[0]
        neg_i = (input_neg_[0][indices_i], input_neg_[1][indices_i], input_neg_[1][indices_i])
        id_ = random.choice([a for a in range(len(indices_i))], size=min_val)
        if neg_chosen_list == None:
            neg_chosen_list = (neg_i[0][id_], neg_i[1][id_], neg_i[1][id_])
        else:
            neg_chosen_list = (np.vstack((neg_i[0][id_], neg_chosen_list[0])), np.vstack((neg_i[1][id_], neg_chosen_list[1])), np.vstack((neg_i[1][id_], neg_chosen_list[2])))
        i += 1
    # alpha2 = (np.min(unique_counts))/np.max(unique_counts)
    # alpha3 = (np.max(unique_counts)) / np.min(unique_counts)
    alpha1 = (np.max(unique_counts) - np.min(unique_counts)) / np.max(unique_counts)
    return neg_chosen_list, unfav, alpha1, np.min(unique_counts), np.max(unique_counts)

# def _cost_sensitive(input_neg):
def _sample_positive_inputs_(input_pos, num_input_pos_sampled):
    """Sample 200 positive inputs.

    :param input_pos:
    :return:
    """
    rg = np.random.default_rng()
    sample = rg.choice(len(input_pos[0]), num_input_pos_sampled)
    input_pos_sampled = (input_pos[0][sample], input_pos[1][sample], input_pos[2][sample])
    # NOTE: Temporally reverted to work with small dataset.
    # returns tuple of the sampled images from input_pos[0]
    # and their respective labels from input_pos[1]
    return input_pos_sampled, sample

def _sample_cross_validation(input_neg, input_pos, folds=5):
    """Sample 200 positive inputs.

    :param input_pos:
    :return:
    """
    kf = model_selection.KFold(n_splits=folds)
    fold_neg = {}
    fold_neg[-1] = input_neg
    for fold, (train_idx, test_idx) in enumerate(kf.split(X=input_neg[0], y=input_neg[1])):
        fold_neg[fold] = (input_neg[0][test_idx], input_neg[1][test_idx], input_neg[1][test_idx])
    fold_pos = {}
    fold_pos[-1] = input_pos
    for fold, (train_idx, test_idx) in enumerate(kf.split(X=input_pos[0], y=input_pos[1])):
        fold_pos[fold] = (input_pos[0][test_idx], input_pos[1][test_idx], input_pos[1][test_idx])
    return fold_neg, fold_pos

def _select_neg_pos_samples(model, input_data):
    y_pred = model.predict(input_data[0], verbose=0)
    pred_labels = np.argmax(y_pred, axis=1)
    y_true = np.argmax(input_data[1], axis=1)

    neg_indices = np.where(y_true != pred_labels)[0]
    pos_indices = np.where(y_true == pred_labels)[0]

    neg_input = (input_data[0][neg_indices], input_data[1][neg_indices], input_data[1][neg_indices])
    pos_input = (input_data[0][pos_indices], input_data[1][pos_indices], input_data[1][pos_indices])
    return neg_input, pos_input



def sample_to_global_size(input_neg_, sample_size):
    neg_labels = np.argmax(input_neg_[1], axis=1)
    unique_val, unique_counts = np.unique(neg_labels, return_counts=True)
    neg_chosen_list = None
    for unique in unique_val:
        indices_i = np.where(neg_labels == unique)[0]
        neg_i = (input_neg_[0][indices_i], input_neg_[1][indices_i], input_neg_[1][indices_i])
        id_ = random.choice([a for a in range(len(indices_i))], size=sample_size)
        if neg_chosen_list == None:
            neg_chosen_list = (neg_i[0][id_], neg_i[1][id_], neg_i[1][id_])
        else:
            neg_chosen_list = (np.vstack((neg_i[0][id_], neg_chosen_list[0])), np.vstack((neg_i[1][id_], neg_chosen_list[1])), np.vstack((neg_i[1][id_], neg_chosen_list[1])))
    return neg_chosen_list
def _parse_test_results_FM_FC(test_images, test_labels, results, unfav=0):
    successes = {}
    failures = {}
    data_all = {}
    dataset_len = len(test_labels)

    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()

        if predicted_label != test_label_index:
            if test_label_index == unfav:
                if test_label_index not in failures:
                    failures[test_label_index] = {}
                if predicted_label not in failures[test_label_index]:
                    failures[test_label_index][predicted_label] = []
                failures[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
                ## todo: saves all the valid datacin the dict
                if test_label_index not in data_all:
                    data_all[test_label_index] = {}
                if predicted_label not in data_all[test_label_index]:
                    data_all[test_label_index][predicted_label] = []
                data_all[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i})
        elif predicted_label == test_label_index and test_label_index == unfav:
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label, "index": i})
            ## todo: saves all the valid datacin the dict
            if test_label_index not in data_all:
                data_all[test_label_index] = {}
            if predicted_label not in data_all[test_label_index]:
                data_all[test_label_index][predicted_label] = []
            data_all[test_label_index][predicted_label].append({"image": test_image, "label": test_label, "index": i})
    return successes, failures, data_all


def _parse_test_results_FM_FCMC(test_images, test_labels, results, unfav=0):
    successes = {}
    failures = {}
    data_all = {}
    dataset_len = len(test_labels)

    for i in range(dataset_len):
        test_image = test_images[i]
        test_label = test_labels[i]
        test_label_index = test_label.argmax()

        result = results[i]
        predicted_label = result.argmax()
        if predicted_label != test_label_index:
            if test_label_index == unfav:
                if test_label_index not in failures:
                    failures[test_label_index] = {}
                if predicted_label not in failures[test_label_index]:
                    failures[test_label_index][predicted_label] = []
                failures[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i}
                )
                ## todo: saves all the valid datacin the dict
                if test_label_index not in data_all:
                    data_all[test_label_index] = {}
                if predicted_label not in data_all[test_label_index]:
                    data_all[test_label_index][predicted_label] = []
                data_all[test_label_index][predicted_label].append(
                    {"image": test_image, "label": test_label, "index": i})
        elif predicted_label == test_label_index:
            if test_label_index not in successes:
                successes[test_label_index] = []
            successes[test_label_index].append({"image": test_image, "label": test_label, "index": i})
            ## todo: saves all the valid datacin the dict
            if test_label_index not in data_all:
                data_all[test_label_index] = {}
            if predicted_label not in data_all[test_label_index]:
                data_all[test_label_index][predicted_label] = []
            data_all[test_label_index][predicted_label].append(
                {"image": test_image, "label": test_label, "index": i})
    return successes, failures, data_all

def _cleanup_dir(path):
    """Clean up given directory.

    Parameters
    ----------
    path : Path
        Path to directory to be cleaned up

    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
def _extract_dataset(dataset):
    images = []
    labels = []
    for result in dataset:
        #print(result)
        image = result["image"]
        label = result["label"]
        images.append(image)
        labels.append(label)
    return images, labels
def _extract_dataset_with_indices(dataset):
    images = []
    labels = []
    indices = []
    for result in dataset:
        #print(result)
        image = result["image"]
        label = result["label"]
        index = result["index"]
        images.append(image)
        labels.append(label)
        indices.append(index)
    return images, labels, indices
def _save_test_result(results, path):
    #print(results)
    images, labels = _extract_dataset(results)
    save_dataset_as_hdf(images, labels, path)
def _save_test_result_with_index(results, path):
    #print(results)
    images, labels, indices = _extract_dataset_with_indices(results)
    save_dataset_as_hdf_with_indices(images, labels, indices, path)
def _save_test_results(results, data_dir):
    for test_label in results:
        # Make directory for each class
        output_dir = data_dir / str(test_label) / "repair.h5"
        output_dir.parent.mkdir(parents=True)
        _save_test_result(results[test_label], output_dir)
def _save_test_results_with_indices(results, data_dir):
    for test_label in results:
        # Make directory for each class
        output_dir = data_dir / str(test_label) / "repair.h5"
        output_dir.parent.mkdir(parents=True)
        _save_test_result_with_index(results[test_label], output_dir)
def _create_merged_dataset(dataset):
    imgs = []
    labels = []
    for label in dataset:
        dataset_per_label = dataset[label]
        for data in dataset_per_label:
            imgs.append(data["image"])
            labels.append(data["label"])
    return imgs, labels

def _create_merged_dataset_with_indices(dataset):
    imgs = []
    labels = []
    indices = []
    for label in dataset:
        dataset_per_label = dataset[label]
        for data in dataset_per_label:
            imgs.append(data["image"])
            labels.append(data["label"])
            indices.append(data["index"])
    return imgs, labels, indices

def _save_positive_results_with_indices(results, data_dir: Path, path):
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    _save_test_results_with_indices(results, output_dir)

    # create all-in-one dataset
    all_images, all_labels, all_indices = _create_merged_dataset_with_indices(results)
    save_dataset_as_hdf_with_indices(all_images, all_labels, all_indices, output_dir / "repair.h5")
def _save_positive_results(results, data_dir: Path, path):
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    _save_test_results(results, output_dir)

    # create all-in-one dataset
    all_images, all_labels = _create_merged_dataset(results)
    save_dataset_as_hdf(all_images, all_labels, output_dir / "repair.h5")
def _save_label_data(data, path):
    summary = {}
    for label in data:
        summary[str(label)] = {
            "repair_priority": 0,
            "prevent_degradation": 0,
        }
    with open(path, "w") as f:
        dict_sorted = sorted(summary.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f, indent=4)
def _save_negative_results(results, data_dir: Path, path):
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    # create each labels repair.h5
    for test_label in results:
        test_label_dir = output_dir / str(test_label)
        test_label_dir.mkdir()
        _save_test_results(results[test_label], test_label_dir)

        # create all-in-one dataset per test label
        images_per_test_label, labels_per_test_label = _create_merged_dataset(results[test_label])
        save_dataset_as_hdf(
            images_per_test_label,
            labels_per_test_label,
            test_label_dir / "repair.h5",
        )
        _save_label_data(results[test_label], test_label_dir / "labels.json")

    # create all-in-one dataset
    all_imgs = []
    all_labels = []
    for labels in results:
        _imgs, _labels = _create_merged_dataset(results[labels])
        all_imgs.extend(_imgs)
        all_labels.extend(_labels)
    save_dataset_as_hdf(all_imgs, all_labels, output_dir / "repair.h5")

def _save_negative_results_with_indices(results, data_dir: Path, path):
    output_dir = data_dir / path
    _cleanup_dir(output_dir)

    # create each labels repair.h5
    for test_label in results:
        test_label_dir = output_dir / str(test_label)
        test_label_dir.mkdir()
        _save_test_results_with_indices(results[test_label], test_label_dir)

        # create all-in-one dataset per test label
        images_per_test_label, labels_per_test_label, indices_per_test_label = _create_merged_dataset_with_indices(results[test_label])
        save_dataset_as_hdf_with_indices(
            images_per_test_label,
            labels_per_test_label,
            indices_per_test_label,
            test_label_dir / "repair.h5",
        )
        _save_label_data(results[test_label], test_label_dir / "labels.json")

    # create all-in-one dataset
    all_imgs = []
    all_labels = []
    all_indices = []
    for labels in results:
        _imgs, _labels, _indices = _create_merged_dataset_with_indices(results[labels])
        all_imgs.extend(_imgs)
        all_labels.extend(_labels)
        all_indices.extend(_indices)
    save_dataset_as_hdf_with_indices(all_imgs, all_labels, all_indices, output_dir / "repair.h5")
def test(model_dir: Path, data_dir: Path, target_data: str, verbose: int = 0, batch_size: int = 32):
    model = load_model_from_tf(model_dir)
    # Load test images and labels
    images, labels = load_dataset_from_hdf(data_dir, target_data)

    # Obtain accuracy as evaluation result of DNN model with test dataset
    score = model.evaluate(images, labels, verbose=verbose, batch_size=batch_size)
    return score
def target(model_dir, data_dir, batch_size):
    # Load DNN model
    model = load_model_from_tf(model_dir)
    # Load test images and labels
    test_images, test_labels = load_dataset_from_hdf(data_dir, "repair.h5")

    # Predict labels from test images
    print("predict")
    results = model.predict(test_images, verbose=0, batch_size=batch_size)
    # Parse and save predict/test results
    print("parse test")
    successes, failures = _parse_test_results(test_images, test_labels, results)
    print("save positive")
    _save_positive_results(successes, data_dir, "positive")
    _save_negative_results(failures, data_dir, "negative")

    _save_label_data(successes, data_dir / "positive/labels.json")
    _save_label_data(failures, data_dir / "negative/labels.json")

def _sample_positive_inputs(input_pos, num_input_pos_sampled):
    """Sample 200 positive inputs.

    :param input_pos:
    :return:
    """
    #rg = np.random.default_rng()
    #sample = rg.choice(len(input_pos[0]), num_input_pos_sampled)
    sample = np.random.choice([i for i in range(input_pos[0].shape[0])], size=num_input_pos_sampled)

    input_pos_sampled = (input_pos[0][sample], input_pos[1][sample], input_pos[2][sample])
    # NOTE: Temporally reverted to work with small dataset.
    # returns tuple of the sampled images from input_pos[0]
    # and their respective labels from input_pos[1]
    return input_pos_sampled, sample
def model_evaluate(model, input_data, verbose, batch_size):
    if hasattr(input_data, "get_generators"):
        images_generator, labels_generator = input_data.get_generators()
        train_generator = tf.data.Dataset.zip(
            (images_generator, labels_generator)
        )  # generates the inputs and labels in batches
        loss, acc = model.evaluate(train_generator, verbose=verbose)
        n = int(np.round(input_data.image_shape[0] * acc))
        #return (loss, acc, n)
    else:
        loss, acc = model.evaluate(
            input_data[0], input_data[1], verbose=verbose, batch_size=batch_size
        )
        n = int(np.round(len(input_data[1]) * acc))
    return (loss, acc, n)
def save_dataset_as_hdf(images, labels, path: Path):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)
def save_dataset_as_hdf_multiple_labels(images, labels, label_name, path: Path):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("images", data=images)
        for i in range(len(label_name)):
            hf.create_dataset(label_name[i], data=labels[i])
def load_dataset_from_hdf(data_dir: Path, target):
    with h5py.File(data_dir / target) as hf:
        test_images = hf["images"][()]
        test_labels = hf["labels"][()]
    return test_images, test_labels

def load_dataset_from_hdf_multiple_labels(data_dir: Path, target):
    with h5py.File(data_dir / target) as hf:
        test_images = hf["images"][()]
        age_labels = hf["age"][()]
        gender_labels = hf["gender"][()]
        race_labels = hf["race"][()]
    return test_images, age_labels, gender_labels, race_labels
def load_dataset_from_hdf_multiple_labels_celebA(data_dir: Path, target):
    with h5py.File(data_dir / target) as hf:
        test_images = hf["images"][()]
        gender_labels = hf["gender"][()]
        young_labels = hf["young"][()]
    return test_images, gender_labels, young_labels

def load_dataset_from_hdf_images_labels_sensitive(data_dir: Path, target):
    with h5py.File(data_dir / target) as hf:
        test_images = hf["images"][()]
        test_labels = hf["labels"][()]
        test_sensitive = hf["sensitive"][()]
    return (test_images, test_labels, test_sensitive)


def save_dataset_as_hdf_with_indices(images, labels, indices, path: Path):
    with h5py.File(path, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)
        hf.create_dataset("indices", data=indices)
def load_dataset_from_hdf_with_indices(data_dir: Path, target):
    with h5py.File(data_dir / target) as hf:
        test_images = hf["images"][()]
        test_labels = hf["labels"][()]
        test_indices = hf["indices"][()]
    return test_images, test_labels, test_indices
def _load_data(data_dir, target):
    data_dir = str(data_dir)
    if data_dir[-1] != "/" and target[0] != "/":
        target = "/" + target
    dataset = EAIDataset(data_dir + target)
    return dataset
def load_repair_data(data_dir):
    return _load_data(data_dir, r"repair.h5")
def load_test_data(data_dir):
        return _load_data(data_dir, r"test.h5")

def _compute_each_gradient(model, x, loss_func, layer):
    # Evaluate grad on neural weights
    with tf.GradientTape(persistent=True) as tape:
        # import pdb; pdb.set_trace()
        logits = model(x[0])  # get the forward pass gradient
        loss_value = loss_func(x[1], logits)
        grad_kernel = tape.gradient(
            loss_value, layer.kernel
        )  # TODO bias?# Evaluate grad on neural weights

    del tape
    return grad_kernel
def _compute_gradient(model, input_data, desc=True, normalise=True):
    # For return
    candidates = []

    # Identify class of loss function
    loss_func = tf.keras.losses.get(model.loss)
    layer_index = len(model.layers) - 1
    layer = model.get_layer(index=layer_index)
    input_neg, input_pos = input_data[0], input_data[1]
    grad_kernel_neg = _compute_each_gradient(model, input_neg, loss_func, layer)
    grad_kernel_pos = _compute_each_gradient(model, input_pos, loss_func, layer)
    list_imp = []

    for j in trange(grad_kernel_neg.shape[1], desc="Computing gradient"):
        for i in range(grad_kernel_neg.shape[0]):
            dl_dw = grad_kernel_neg[i][j]
            dl_dw_pos = grad_kernel_pos[i][j]
            # Append data tuple
            # (layer, i, j) is for identifying neural weight
            #dl_dw_ratio = np.abs(dl_dw)/(1+np.abs(dl_dw_pos))
            dl_dw_ratio = abs(np.abs(dl_dw) - np.abs(dl_dw_pos))
            #dl_dw_ratio = abs(np.abs(dl_dw_pos)-np.abs(dl_dw))# abs(np.abs(dl_dw) - np.abs(dl_dw_pos))

            # dl_dw_ratio = abs(dl_dw - dl_dw_pos)
            # candidates.append([layer_index, i, j, dl_dw_ratio])
            list_imp.append(dl_dw_ratio)
    if normalise:
        list_imp = [float(i) / sum(list_imp) for i in list_imp]
    indices = 0
    for j in trange(grad_kernel_neg.shape[1], desc="Computing gradient"):
        for i in range(grad_kernel_neg.shape[0]):
            candidates.append([layer_index, i, j, list_imp[indices]])
            indices += 1
    # Sort candidates in order of grad loss
    candidates.sort(key=lambda tup: tup[3], reverse=desc)

    return candidates  # , grad_kernel, grad_kernel_pos, loss_value, loss_value_pos


def compute_gradian_loss(model, input_data):
    # Identify class of loss function
    loss_func = tf.keras.losses.get(model.loss)
    layer_index = len(model.layers) - 1
    layer = model.get_layer(index=layer_index)
    input_0, input_1 = input_data[0], input_data[1]
    max_lenth = input_0[0].shape[0]
    if input_0[0].shape[0] > input_1[0].shape[0]:
        max_lenth = input_1[0].shape[0]

    # Evaluate grad on neural weights
    with tf.GradientTape(persistent=True) as tape:
        # import pdb; pdb.set_trace()
        logits = model(input_0[0])  # get the forward pass gradient
        loss_value = loss_func(input_0[1], logits)
        grad_kernel = tape.gradient(
            loss_value, layer.kernel
        )  # TODO bias?# Evaluate grad on neural weights

    del tape

    # Evaluate grad on neural weights
    with tf.GradientTape(persistent=True) as tape:
        # import pdb; pdb.set_trace()
        ## todo: positive inputs
        logits_1 = model(input_1[0])  # get the forward pass gradient
        loss_value_1 = loss_func(input_1[1], logits_1)

        grad_kernel_1 = tape.gradient(
            loss_value_1, layer.kernel
        )  # TODO bias?# Evaluate grad on neural weights

        # print(grad_kernel.shape, input_neg[0].shape, input_neg[1].shape)
    del tape
    return grad_kernel, grad_kernel_1


def _compute_gradient_pos_neg(model, input_neg, input_pos, desc=True, normalise=True, normalise_function=np.max):
    # For return
    candidates = []

    # Identify class of loss function
    loss_func = tf.keras.losses.get(model.loss)
    layer_index = len(model.layers) - 1
    layer = model.get_layer(index=layer_index)
    # input_neg, input_pos = input_data[0], input_data[1]

    grad_kernel_neg_0, grad_kernel_neg_1 = compute_gradian_loss(model, input_neg)
    grad_kernel_pos_0, grad_kernel_pos_1 = compute_gradian_loss(model, input_pos)
    list_imp = []

    for j in trange(grad_kernel_neg_0.shape[1], desc="Computing gradient"):
        for i in range(grad_kernel_neg_0.shape[0]):
            dl_dw_neg_0 = grad_kernel_neg_0[i][j]
            dl_dw_neg_1 = grad_kernel_neg_1[i][j]

            dl_dw_pos_0 = grad_kernel_pos_0[i][j]
            dl_dw_pos_1 = grad_kernel_pos_1[i][j]

            # Append data tuple
            # (layer, i, j) is for identifying neural weight
            dl_dw_ratio_0 = np.abs(dl_dw_neg_0) / (1 + np.abs(dl_dw_pos_0))
            dl_dw_ratio_1 = np.abs(dl_dw_neg_1) / (1 + np.abs(dl_dw_pos_1))
            #dl_dw_ratio = abs(dl_dw_ratio_0 - dl_dw_ratio_1)
            dl_dw_ratio = abs(dl_dw_ratio_1 - dl_dw_ratio_0) # abs(dl_dw_ratio_0 - dl_dw_ratio_1)
            # dl_dw_ratio = abs(dl_dw - dl_dw_pos)
            # candidates.append([layer_index, i, j, dl_dw_ratio])
            list_imp.append(dl_dw_ratio)
    if normalise:
        list_imp = [float(i) / normalise_function(list_imp) for i in list_imp]

    indices = 0
    for j in trange(grad_kernel_neg_0.shape[1], desc="Computing gradient"):
        for i in range(grad_kernel_neg_0.shape[0]):
            candidates.append([layer_index, i, j, list_imp[indices]])
            indices += 1
    # Sort candidates in order of grad loss
    candidates.sort(key=lambda tup: tup[3], reverse=desc)

    return candidates
def _compute_each_forward_impact_neg_pos(weight, activations, w):
    layer_index = weight[0]
    neural_weight_i = weight[1]
    neural_weight_j = weight[2]

    if layer_index < 1:
        raise IndexError(f"Not found previous layer: {layer_index!r}")
    activations_0, activations_1 = activations
    o_i_0 = activations_0[0][neural_weight_i]  # TODO correct?
    o_i_1 = activations_1[0][neural_weight_i]  # TODO correct?
    #w_0, w_1 = w
    # Evaluate the neural weight
    w_ij = w[neural_weight_i][neural_weight_j]
    #w_ij_1 = w_1[neural_weight_i][neural_weight_j]

    return np.abs(o_i_0 * w_ij), np.abs(o_i_1 * w_ij)

def _compute_each_forward_impact(weight, activations, w):
    layer_index = weight[0]
    neural_weight_i = weight[1]
    neural_weight_j = weight[2]

    if layer_index < 1:
        raise IndexError(f"Not found previous layer: {layer_index!r}")

    o_i = activations[0][neural_weight_i]  # TODO correct?

    # Evaluate the neural weight
    w_ij = w[neural_weight_i][neural_weight_j]

    return np.abs(o_i * w_ij)
def _compute_forward_impact(model, input_neg,  candidates, num_grad):
    pool = {}
    layer_index = candidates[0][0]
    _num_grad = num_grad if num_grad < len(candidates) else len(candidates)
    previous_layer = model.get_layer(index=layer_index - 1)
    target_layer = model.get_layer(index=layer_index)

    # Evaluate activation value of the corresponding neuron
    # in the previous layer
    get_activations = K.function([model.input], previous_layer.output)
    activations = get_activations(input_neg[0])
    # Evaluate the neuron weight
    w = K.eval(target_layer.kernel)

    for num in trange(_num_grad, desc="Computing forward impact"):
        layer_index, i, j, grad_loss = candidates[num]
        fwd_imp = _compute_each_forward_impact(
            model, input_neg, [layer_index, i, j], activations, w
        )
        pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
    return pool

def _compute_forward_impact_neg_pos(model, input_neg, input_pos,  candidates, num_grad, normalise=True, normalise_function=np.max):
    pool = {}
    layer_index = candidates[0][0]
    _num_grad = num_grad if num_grad < len(candidates) else len(candidates)
    previous_layer = model.get_layer(index=layer_index - 1)
    target_layer = model.get_layer(index=layer_index)



    neg_0, neg_1 = select_subgroups_from_input(input_neg)
    pos_0, pos_1 = select_subgroups_from_input(input_pos)
    # Evaluate activation value of the corresponding neuron
    # in the previous layer
    get_activations = K.function([model.input], previous_layer.output)
    activations_neg_0 = get_activations(neg_0[0])
    activations_neg_1 = get_activations(neg_1[0])

    activations_pos_0 = get_activations(pos_0[0])
    activations_pos_1 = get_activations(pos_1[0])
    # Evaluate the neuron weight
    w = K.eval(target_layer.kernel)

    grad_loss_diff_list, fwd_imp_diff_list, sum_list = [], [], []

    for num in trange(_num_grad, desc="Computing forward impact"):
        layer_index, i, j, grad_loss_diff = candidates[num]
        fwd_imp_neg_0, fwd_imp_neg_1 = _compute_each_forward_impact_neg_pos([layer_index, i, j], (activations_neg_0, activations_neg_1), w
        )
        fwd_imp_pos_0, fwd_imp_pos_1 = _compute_each_forward_impact_neg_pos([layer_index, i, j], (activations_pos_0, activations_pos_1), w
        )
        fwd_imp_diff = abs(fwd_imp_neg_0/(1+fwd_imp_pos_0) - fwd_imp_neg_1/(1+fwd_imp_pos_1))
        pool[num] = [layer_index, i, j, grad_loss_diff, fwd_imp_diff]
        grad_loss_diff_list.append(grad_loss_diff)
        fwd_imp_diff_list.append(fwd_imp_diff)
        sum_list.append(grad_loss_diff*fwd_imp_diff)
    if normalise:
        grad_loss_diff_list = [float(i) / normalise_function(grad_loss_diff_list) for i in grad_loss_diff_list]
        fwd_imp_diff_list = [float(i) / normalise_function(fwd_imp_diff_list) for i in fwd_imp_diff_list]
        sum_list = [float(i) / normalise_function(sum_list) for i in sum_list]
    return pool, grad_loss_diff_list, fwd_imp_diff_list, sum_list

def select_subgroups_from_input(input_neg_):
    neg_labels = np.argmax(input_neg_[1], axis=1)
    unique_val, unique_counts = np.unique(neg_labels, return_counts=True)
    min_val = np.min(unique_counts)
    neg_chosen_list = None
    indices_0 = np.where(neg_labels == 0)[0]
    neg_0 = (input_neg_[0][indices_0], input_neg_[1][indices_0], input_neg_[1][indices_0])

    indices_1 = np.where(neg_labels == 1)[0]
    neg_1 = (input_neg_[0][indices_1], input_neg_[1][indices_1], input_neg_[1][indices_1])

    #ratio_ = ((1 + np.max(unique_counts)) - (1 + np.min(unique_counts))) / (1 + np.min(unique_counts))

    #unfav = 0
    if len(indices_0) < len(indices_1):
        unfav = 1
    return neg_0, neg_1

def select_pos_neg(model, input_data):
    true_labels = np.argmax(input_data[1], axis=1)

    pred = model.predict(input_data[0], verbose=0)
    pred_labels = np.argmax(pred, axis=1)
    indices_neg = np.where(pred_labels != true_labels)[0]
    indices_pos = np.where(pred_labels == true_labels)[0]
    input_neg = (input_data[0][indices_neg], input_data[1][indices_neg], true_labels[indices_neg])
    input_pos = (input_data[0][indices_pos], input_data[1][indices_pos], true_labels[indices_pos])

    return input_neg, input_pos

def _get_bias_weight_for(y_true, y_pred, attrib=0):
    N = len(y_true)
    F = (y_true == attrib).sum()
    C = (y_true == y_pred).sum()
    M_ = (y_true != y_pred).sum()
    FC = ((y_true == y_pred) & (y_true == attrib)).sum()
    FM = ((y_true != y_pred) & (y_pred == attrib)).sum()

    W_FC = (F / N) * (FC / N) / (C / N)
    W_FM = (F / N) * (FM / N) / (M_ / N)

    return W_FC, W_FM

def _get_unfav(y_true, y_pred):
    # keys = list(fitness_dict.keys())
    sensitive_unique = np.unique(y_true)
    sensitive_unique.sort()


    # Count_0, Count_1 = 0, 0
    # P_obs_0, P_obs_1 = 0, 0
    N = len(y_true)
    # unpriveledge = sensitive_unique[0]
    M = (y_true == 1).sum()
    F = (y_true == 0).sum()

    C = (y_true == y_pred).sum()
    M_ = (y_true != y_pred).sum()

    MC = ((y_true == y_pred) & (y_true == 1)).sum()
    FC = ((y_true == y_pred) & (y_pred == 0)).sum()
    MM = ((y_true != y_pred) & (y_pred == 0)).sum()
    FM = ((y_true != y_pred) & (y_pred == 1)).sum()

    # W_MC = ((M / N) * (MC / N)) / (C / N)
    # W_MM = (M / N) * (MM / N) / (M_ / N)
    # W_FC = (F / N) * (FC / N) / (C / N)
    # W_FM = (F / N) * (FM / N) / (M_ / N)

    W_P_exp_0 = (F / N) * (C / N)
    W_P_obs_0 = (FC / N)

    W_P_exp_1 = ((M / N) * (C / N))
    W_P_obs_1 = (MC / N)

    if W_P_exp_0 > W_P_obs_0:
        unpriveledge = 0
    elif W_P_exp_1 > W_P_obs_1:
        unpriveledge = 1
    else:
        #print('privilage unknown...')
        unpriveledge = 0
    # W_F = W_P_exp_0/W_P_obs_0
    # W_M = W_P_exp_1/W_P_obs_1

    return unpriveledge

def _get_subgroup_names_with_sensitive(y_true, y_pred, sensitive_attributes):
    # keys = list(fitness_dict.keys())
    sensitive_unique = np.unique(sensitive_attributes)
    sensitive_unique.sort()
    sensitive_0, sensitive_1 = sensitive_unique[0], sensitive_unique[1]
    # Count_0, Count_1 = 0, 0
    # P_obs_0, P_obs_1 = 0, 0
    N = len(y_true)
    #unpriveledge = sensitive_unique[0]

    S1 = (y_true == sensitive_1).sum()
    S0 = (y_true == sensitive_0).sum()

    C1 = (y_true == 1).sum()
    C0 = (y_true == 0).sum()

    C = (y_true == y_pred).sum()

    CC1 = ((y_true == y_pred) & (y_true == 1)).sum()
    CC0 = ((y_true == y_pred) & (y_pred == 0)).sum()

    W_P_exp_0 = (C0 / N) * (C / N)
    W_P_obs_0 = (CC0 / N)

    W_P_exp_1 = ((C1 / N) * (C / N))
    W_P_obs_1 = (CC1 / N)

    if W_P_exp_0 > W_P_obs_0:
        unpriveledge = 0
    elif W_P_exp_1 > W_P_obs_1:
        unpriveledge = 1
    else:
        unpriveledge = 0


    S11 = ((y_true == 1) & (sensitive_attributes == sensitive_1)).sum()
    S10 = ((y_true == 1) & (sensitive_attributes == sensitive_0)).sum()

    S00 = ((y_true == 0) & (sensitive_attributes == sensitive_0)).sum()
    S01 = ((y_true == 0) & (sensitive_attributes == sensitive_1)).sum()

    W_S11 = ((S1 / N) * (C1 / N)) / (S11 / N)

    W_S10 = ((S0 / N) * (C1 / N)) / (S10 / N)
    W_S00 = ((S0 / N) * (C0 / N)) / (S00 / N)
    W_S01 = ((S1 / N) * (C0 / N)) / (S01 / N)

    if W_S11 > W_S11:
        unfav_1 = sensitive_1
    else:
        unfav_1 = sensitive_0

    if W_S00 > W_S01:
        unfav_0 = sensitive_0
    else:
        unfav_0 = sensitive_1

    return unpriveledge, unfav_0, unfav_1, W_S00, W_S01, W_S10, W_S11, sensitive_0, sensitive_1

def _get_subgroup_names_as_binary(y_true, y_pred):
    # keys = list(fitness_dict.keys())
    sensitive_unique = np.unique(y_true)
    sensitive_unique.sort()


    # Count_0, Count_1 = 0, 0
    # P_obs_0, P_obs_1 = 0, 0
    N = len(y_true)
    unpriveledge = sensitive_unique[0]

    M = (y_true == 1).sum()
    F = (y_true == 0).sum()

    C = (y_true == y_pred).sum()
    M_ = (y_true != y_pred).sum()

    MC = ((y_true == y_pred) & (y_true == 1)).sum()
    FC = ((y_true == y_pred) & (y_pred == 0)).sum()
    MM = ((y_true != y_pred) & (y_pred == 0)).sum()
    FM = ((y_true != y_pred) & (y_pred == 1)).sum()

    W_MC = ((M / N) * (MC / N)) / (C / N)
    W_MM = ((M / N) * (MM / N)) / (M_ / N)
    W_FC = ((F / N) * (FC / N)) / (C / N)
    W_FM = ((F / N) * (FM / N)) / (M_ / N)

    W_P_exp_0 = (F / N) * (C / N)
    W_P_obs_0 = (FC / N)

    W_P_exp_1 = ((M / N) * (C / N))
    W_P_obs_1 = (MC / N)



    #print('Weights: W_MC: {}, W_MM: {}, W_FC: {}, W_FM: {}'.format(W_MC, W_MM, W_FC, W_FM))
    #
    #
    #
    # Y_1 = 0
    # prev_class = np.unique(y_true)[0]
    #
    # indices_0, indices_1 = [], []
    # for i in range(len(y_true)):
    #     if y_true[i] == 0:
    #         indices_0.append(i)
    #         Count_0 += 1
    #         if y_true[i] == y_pred[i]:
    #             P_obs_0 += 1
    #             Y_1 += 1
    #     else:
    #         indices_1.append(i)
    #         Count_1 += 1
    #         if y_true[i] == y_pred[i]:
    #             P_obs_1 += 1
    #             Y_1 += 1
    # P_exp_0 = (Count_0 / N) * (Y_1 / N)
    # P_exp_1 = (Count_1 / N) * (Y_1 / N)
    # P_obs_0 = P_obs_0 / N
    # P_obs_1 = P_obs_1 / N
    if W_P_exp_0 > W_P_obs_0:
        unpriveledge = 0
    elif W_P_exp_1 > W_P_obs_1:
        unpriveledge = 1
    else:
        #print('privilage unknown...')
        unpriveledge = 0
    W_F = W_P_exp_0/W_P_obs_0
    W_M = W_P_exp_1/W_P_obs_1
    #print('P_exp_A: ', W_P_exp_0, 'P_obs_A: ', W_P_obs_0)
    #print('P_exp_B: ', W_P_exp_1, 'P_obs_B: ', W_P_obs_1)
    #print('unpriveledge: ', unpriveledge)
    return unpriveledge, W_M, W_F, W_MC, W_MM, W_FC, W_FM, MC, MM, FC, FM
def _parse_results_sample_negative(test_images, test_labels, pred_labels, unfav):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    neg_indices_all = np.where(y_true_argmax != y_pred_argmax)[0]
    pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    # indices_f = np.where(y_true_argmax == 0)[0]
    # indices_m = np.where(y_true_argmax == 1)[0]

    neg_indices = np.where((y_true_argmax != y_pred_argmax) & y_true_argmax==unfav)[0]

    min_size = len(neg_indices_all)
    sample_pos = np.random.choice(pos_indices, size=min_size)
    input_neg = (test_images[neg_indices], test_labels[neg_indices],
                 np.argmax(test_labels[neg_indices], axis=1).reshape(-1, 1))
    input_pos = (test_images[sample_pos], test_labels[sample_pos],
                 np.argmax(test_labels[sample_pos], axis=1).reshape(-1, 1))

    input_both = (np.vstack((input_neg[0], input_pos[0])),
                  np.vstack((input_neg[1], input_pos[1])),
                  np.vstack((input_neg[2], input_pos[2])))

    return input_neg, input_pos, input_both

def _parse_results_sample_negative_sensitive(test_images, test_labels, pred_labels, sensitive_labels, unfav):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    neg_indices_all = np.where(y_true_argmax != y_pred_argmax)[0]
    pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    # indices_f = np.where(y_true_argmax == 0)[0]
    # indices_m = np.where(y_true_argmax == 1)[0]

    neg_indices = np.where((y_true_argmax != y_pred_argmax) & y_true_argmax==unfav)[0]

    min_size = len(neg_indices_all)
    sample_pos = np.random.choice(pos_indices, size=min_size)
    input_neg = (test_images[neg_indices], test_labels[neg_indices], sensitive_labels[neg_indices])
    input_pos = (test_images[sample_pos], test_labels[sample_pos], sensitive_labels[sample_pos])

    sensi_both_indices = np.hstack((neg_indices, sample_pos))
    sensi_both = sensitive_labels[sensi_both_indices]

    input_both = (np.vstack((input_neg[0], input_pos[0])),
                  np.vstack((input_neg[1], input_pos[1])),
                  sensi_both)

    return input_neg, input_pos, input_both


def _parse_results_sampling_sensitive(test_images, test_labels, pred_labels, sensitive_labels, unfav_0=0, unfav_1=1):

    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #sensitive_argmax = np.argmax(sensitive_labels, axis=1)

    neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    sample_pos = np.random.choice([i for i in pos_indices], size=len(neg_indices))

    #print(sensitive_labels[neg_indices].shape, sensitive_labels[sample_pos].shape)

    sampled_both = np.hstack((neg_indices, sample_pos))  #(np.vstack((neg_indices, sample_pos)))
    #print(input_both[1].shape, input_both[2].shape, y_true_sampled.shape, input_both[2])
    indices_unfav_0 = np.where((y_true_argmax == 0) & (sensitive_labels == unfav_0))[0]
    indices_fav_0 = np.where((y_true_argmax == 0) & (sensitive_labels != unfav_0))[0]

    #indices_unfav_0 = [i for i in indices_unfav_0 if i in sampled_both]

    indices_unfav_1 = np.where((y_true_argmax == 1) & (sensitive_labels == unfav_1))[0]
    indices_fav_1 = np.where((y_true_argmax == 1) & (sensitive_labels != unfav_1))[0]

    indices_unfav = np.hstack((indices_unfav_0, indices_unfav_1)) # np.vstack((indices_unfav_0, indices_unfav_1))
    indices_fav = np.hstack((indices_fav_0, indices_fav_1))  # np.vstack((indices_fav_0, indices_fav_1))

    indices_unfav = [i for i in indices_unfav if i in sampled_both]
    indices_fav = [i for i in indices_fav if i in sampled_both]
    sampled_both = [i for i in sampled_both]
    sens_labels = np.array([i for i in sensitive_labels])[sampled_both]

    #print('sampled_both: ', sampled_both)

    #print(test_images.shape, test_labels.shape, sensitive_labels.shape)

    input_both = (test_images[sampled_both], test_labels[sampled_both], sens_labels)

    input_unfav = (test_images[indices_unfav], test_labels[indices_unfav], sensitive_labels[indices_unfav])
    input_fav = (test_images[indices_fav], test_labels[indices_fav], sensitive_labels[indices_fav])


    # input_both = (np.vstack((input_unfav[0], input_fav[0])), np.vstack((input_unfav[1], input_fav[1])),
    #               np.vstack((input_unfav[2].reshape(-1,1), input_fav[2].reshape(-1,1))))
    return input_unfav, input_fav, input_both

def _parse_results_uniform_sampling(test_images, test_labels, pred_labels):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    #pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    indices_f = np.where(y_true_argmax == 0)[0]
    indices_m = np.where(y_true_argmax == 1)[0]

    min_size = len(indices_f)
    if len(indices_f) > len(indices_m):
        min_size = len(indices_m)

    sample_M = np.random.choice(indices_m, size=min_size)
    sample_F = np.random.choice(indices_f, size=min_size)


    samples_ = [i for i in sample_F]
    for i in sample_M:
        samples_.append(i)

    y_true_sampled = y_true_argmax[samples_]
    y_pred_sampled = y_pred_argmax[samples_]

    test_images_sampled = test_images[samples_]
    test_labels_sampled = test_labels[samples_]

    neg_indices = np.where(y_true_sampled != y_pred_sampled)[0]
    pos_indices = np.where(y_true_sampled == y_pred_sampled)[0]

    input_neg = (test_images_sampled[neg_indices], test_labels_sampled[neg_indices],
                 np.argmax(test_labels_sampled[neg_indices], axis=1).reshape(-1, 1))
    input_pos = (test_images_sampled[pos_indices], test_labels_sampled[pos_indices],
                 np.argmax(test_labels_sampled[pos_indices], axis=1).reshape(-1, 1))

    input_both = (np.vstack((input_neg[0], input_pos[0])),
                  np.vstack((input_neg[1], input_pos[1])),
                  np.vstack((input_neg[2], input_pos[2])))

    return input_neg, input_pos, input_both

def _parse_results_uniform_sampling_positive_same(test_images, test_labels, pred_labels):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    #pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    MC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 1))[0]
    FC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 0))[0]

    MM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 1))[0]
    FM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 0))[0]

    pos_size = len(MC_Indices)
    if len(MC_Indices) > len(FC_Indices):
        pos_size = len(MC_Indices)

    sample_MC = np.random.choice(MC_Indices, size=pos_size)
    sample_FC = np.random.choice(FC_Indices, size=pos_size)
    #sample_MM = np.random.choice(MM_Indices, size=size_mm)
    #sample_FM = np.random.choice(FM_Indices, size=size_fm)

    input_neg = (np.vstack((test_images[FM_Indices], test_images[MM_Indices])),
                 np.vstack((test_labels[FM_Indices], test_labels[MM_Indices])),
                 np.vstack((np.argmax(test_labels[FM_Indices], axis=1).reshape(-1,1),
                            np.argmax(test_labels[MM_Indices], axis=1).reshape(-1,1))))

    input_pos = (np.vstack((test_images[sample_FC], test_images[sample_MC])),
                 np.vstack((test_labels[sample_FC], test_labels[sample_MC])),
                 np.vstack((np.argmax(test_labels[sample_FC], axis=1).reshape(-1, 1),
                            np.argmax(test_labels[sample_MC], axis=1).reshape(-1, 1))))

    input_both = (np.vstack((input_neg[0], input_pos[0])), np.vstack((input_neg[1], input_pos[1])))

    return input_neg, input_pos, input_both

def _parse_results_uniform_sampling_positive(test_images, test_labels, pred_labels, size_mc, size_fc):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)




    MC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 1))[0]
    FC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 0))[0]

    MM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 1))[0]
    FM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 0))[0]

    sample_MC = np.random.choice(MC_Indices, size=size_mc)
    sample_FC = np.random.choice(FC_Indices, size=size_fc)
    #sample_MM = np.random.choice(MM_Indices, size=size_mm)
    #sample_FM = np.random.choice(FM_Indices, size=size_fm)

    input_neg = (np.vstack((test_images[FM_Indices], test_images[MM_Indices])),
                 np.vstack((test_labels[FM_Indices], test_labels[MM_Indices])),
                 np.vstack((np.argmax(test_labels[FM_Indices], axis=1).reshape(-1,1),
                            np.argmax(test_labels[MM_Indices], axis=1).reshape(-1,1))))

    input_pos = (np.vstack((test_images[sample_FC], test_images[sample_MC])),
                 np.vstack((test_labels[sample_FC], test_labels[sample_MC])),
                 np.vstack((np.argmax(test_labels[sample_FC], axis=1).reshape(-1, 1),
                            np.argmax(test_labels[sample_MC], axis=1).reshape(-1, 1))))

    input_both = (np.vstack((input_neg[0], input_pos[0])), np.vstack((input_neg[1], input_pos[1])))

    return input_neg, input_pos, input_both

