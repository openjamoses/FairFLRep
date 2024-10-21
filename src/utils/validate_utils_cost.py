import csv
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from src.dnn_testing.eAI.utils.arachne_fairness_validation import FairArachne2
from src.dnn_testing.eAI.utils.evaluate_utils import _fairness_sub
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K  # noqa: N812
from tqdm import tqdm, trange
from pathlib import Path
from src.dnn_testing.eAI.utils import data_util
from src.dnn_testing.eAI.utils.fault_localization_validation_cost import FaultLocalization


class Localization:
    def __init__(self, model_dir, input_neg, input_pos, input_both, unfav=0, based_on=0, sample_global=False):
        self.unfav = unfav
        self.based_on = based_on
        self.model_dir = Path(model_dir)
        self.model = data_util.load_model_from_tf(Path(model_dir))
        self.target_layer = None
        if self.target_layer == len(self.model.layers) - 1:
            self.target_layer = None

        neg_labels = np.argmax(input_neg[1], axis=1)
        unique_val, unique_counts = np.unique(neg_labels, return_counts=True)
        self.global_sample = np.min(unique_counts)
        pos_labels = np.argmax(input_pos[1], axis=1)
        unique_val, unique_counts = np.unique(pos_labels, return_counts=True)
        if np.min(unique_counts) < self.global_sample:
            self.global_sample = np.min(unique_counts)
        if sample_global:
            self.input_neg = data_util.sample_to_global_size(input_neg, self.global_sample)
            self.input_pos = data_util.sample_to_global_size(input_pos, self.global_sample)
        else:
            self.input_neg_, _, alpha1, alpha2, alpha3 = data_util.sample_negative(input_neg)
            self.input_pos_, _, alpha1, alpha2, alpha3 = data_util.sample_negative(input_pos)
        self.input_both, _, _, _, _ = data_util.sample_negative(input_both)

        self.input_neg = input_neg
        self.input_pos = input_pos

    def _compute_each_gradient(self, model, x, x2, layer_index):
        # Identify class of loss function
        loss_func = tf.keras.losses.get(model.loss)
        layer = model.get_layer(index=layer_index)
        # Evaluate grad on neural weights
        with tf.GradientTape(persistent=True) as tape:  # persistent=True
            # import pdb; pdb.set_trace()
            logits = model(x[0])  # get the forward pass gradient
            logits_x2 = model(x2[0])  # get the forward pass gradient
            loss_value = loss_func(x[1], logits)
            loss_value_x2 = loss_func(x2[1], logits_x2)
        grad_kernel = tape.gradient(
            loss_value, layer.kernel
        )  # TODO bias?# Evaluate grad on neural weights
        grad_kernel_x2 = tape.gradient(
            loss_value_x2, layer.kernel
        )  # TODO bias?# Evaluate grad on neural weights
        # del tape
        return grad_kernel, grad_kernel_x2

    def _agregate_based_on_fairness_notion(self, val_0, val_1, based_on=0):
        if based_on > 2:
            return ((val_0) / (1 + val_1))
        else:
            if self.unfav == 0:
                return ((val_0) / (1 + val_1))  # + self.alpha_3/self.alpha_2
            else:
                return ((val_1) / (1 + val_0))  # + self.alpha_3/self.alpha_2

    def _compute_gradient(self, model, input_data_, based_on=0, desc=True, is_sampled=False, normalise=False,
                          normalise_function=np.max):
        # For return
        candidates = []
        if is_sampled:
            input_data = input_data_  # data_util.sample_negative(input_data_)
        else:
            input_data, _, self.alpha_, self.alpha2, self.alpha3 = data_util.sample_negative(input_data_)
        input_data = input_data_
        input_0, input_1 = data_util.select_subgroups_from_input(input_data)
        # self.unfav = unfav
        layer_index = len(model.layers) - 1
        self.layer_index = layer_index
        grad_kernel_0, grad_kernel_1 = self._compute_each_gradient(model, input_0, input_1, layer_index)
        list_imp = []
        dl_dw0_list = []
        dl_dw1_list = []

        for j in trange(grad_kernel_0.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel_0.shape[0]):
                dl_dw0 = grad_kernel_0[i][j]
                dl_dw1 = grad_kernel_1[i][j]
                dl_dw0_list.append(dl_dw0)
                dl_dw1_list.append(dl_dw1)

        if normalise:
            dl_dw0_list = [float(i) / normalise_function(dl_dw0_list) for i in dl_dw0_list]
            dl_dw1_list = [float(i) / normalise_function(dl_dw1_list) for i in dl_dw1_list]
        index_ = 0
        self.alpha_1 = self.alpha_ * np.max(dl_dw0_list)
        self.alpha_2 = self.alpha2 / self.alpha3 * np.max(dl_dw1_list)
        self.alpha_3 = self.alpha3 / self.alpha2 * np.max(dl_dw1_list)
        dl_dw0_pools = {}
        dl_dw2_pools = {}
        list_sorted = []
        list_sorted1 = []
        list_sorted2 = []
        for j in trange(grad_kernel_0.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel_0.shape[0]):
                dl_dw0 = dl_dw0_list[index_]
                dl_dw1 = dl_dw1_list[index_]
                dl_dw = self._agregate_based_on_fairness_notion(np.abs(dl_dw0), np.abs(dl_dw1), based_on=based_on)

                list_sorted1.append([layer_index, i, j, np.abs(dl_dw0)])
                list_sorted2.append([layer_index, i, j, np.abs(dl_dw1)])
                list_sorted.append([layer_index, i, j, dl_dw0, dl_dw1, dl_dw])
                index_ += 1
        list_sorted.sort(key=lambda tup: tup[5], reverse=desc)
        # list_sorted11, list_sorted22 = zip(*sorted(zip(list_sorted1, list_sorted2), reverse=True))

        data_candidate_dict = {}
        for num in range(len(list_sorted)):
            layer_index, i, j, dl_dw0, dl_dw1, _ = list_sorted[num]
            dl_dw = self._agregate_based_on_fairness_notion(np.abs(dl_dw0), np.abs(dl_dw1), based_on)
            candidates.append([layer_index, i, j, np.abs(dl_dw)])
            list_imp.append([layer_index, i, j, np.abs(dl_dw), dl_dw0, dl_dw1])
            if i in data_candidate_dict.keys():
                data_candidate_dict[i][j] = np.abs(dl_dw)
            else:
                data_candidate_dict[i] = {}
                data_candidate_dict[i][j] = np.abs(dl_dw)
        # Sort candidates in order of grad loss
        candidates.sort(key=lambda tup: tup[3], reverse=desc)
        # list_imp.sort(key=lambda tup: tup[3], reverse=desc)
        return list_imp, candidates, data_candidate_dict

    def _compute_each_forward_impact(self, weight, activations, w):
        layer_index = weight[0]
        neural_weight_i = weight[1]
        neural_weight_j = weight[2]
        if layer_index < 1:
            raise IndexError(f"Not found previous layer: {layer_index!r}")
        o_i = activations[0][neural_weight_i]  # TODO correct?
        # Evaluate the neural weight
        w_ij = w[neural_weight_i][neural_weight_j]
        return np.abs(o_i * w_ij)

    def _forward_impact(self, model_, input_, layer_index, list_candidate, _num_grad, normalise=False,
                        normalise_function=np.sum):
        model_ = data_util.load_model_from_tf(self.model_dir)
        model = self._reshape_target_model(model_, input_)
        previous_layer = model.get_layer(index=layer_index - 1)
        target_layer = model.get_layer(index=layer_index)
        # Evaluate activation value of the corresponding neuron
        get_activations_0 = K.function([model.input], previous_layer.output)
        activations_0 = get_activations_0(input_[0])
        w0 = K.eval(target_layer.kernel)

        list_fwd_imp_0 = []
        for num in range(len(list_candidate)):  # trange(_num_grad, desc="Computing forward impact"):
            layer_index, i, j, grad_loss, _, _ = list_candidate[num]
            fwd_imp_0 = self._compute_each_forward_impact([layer_index, i, j], activations_0, w0)
            list_fwd_imp_0.append(fwd_imp_0)
        if normalise:
            list_fwd_imp_0 = [float(i) / normalise_function(list_fwd_imp_0) for i in list_fwd_imp_0]
        return list_fwd_imp_0

    def _compute_forward_impact_single_new(self, model, input_data_, candidates_, list_candidate, data_candidate_dict,
                                           num_grad, is_sampled=False, desc=True, based_on=0):
        pool = {}
        list_pool = []
        # candidates, candidates1, candidates2 = candidates_
        layer_index = candidates_[0][0]
        _num_grad = num_grad if num_grad < len(candidates_) else len(candidates_)
        if is_sampled:
            input_data = input_data_  # data_util.sample_negative(input_data_)
        else:
            input_data, _, self.alpha_, self.alpha2, self.alpha3 = data_util.sample_negative(input_data_)
        input_data = input_data_
        input_0, input_1 = data_util.select_subgroups_from_input(input_data)
        fwd_imp_0_list = self._forward_impact(model, input_0, layer_index, list_candidate, _num_grad)
        fwd_imp_1_list = self._forward_impact(model, input_1, layer_index, list_candidate, _num_grad)
        list_pool_sorted = []
        pl_data = {}
        # pl_1 = {}
        for num in range(len(list_candidate)):
            layer_index, i, j, grad_loss, _, _ = list_candidate[num]
            if not i in pl_data.keys():
                pl_data[i] = {}
            pl_data[i][j] = (fwd_imp_0_list[num], fwd_imp_1_list[num])

        list_candidate.sort(key=lambda tup: tup[3], reverse=desc)
        grad_lst = []
        fwd_imp_lst = []
        for num in trange(_num_grad, desc="Computing forward impact"):
            layer_index, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
            fwd_imp_0, fwd_imp_1 = pl_data[i][j]  # pl_0[num], pl_1[num]
            fwd_imp = self._agregate_based_on_fairness_notion(fwd_imp_0, fwd_imp_1, based_on)
            list_pool_sorted.append([layer_index, i, j, grad_loss, fwd_imp])
            list_pool.append((layer_index, i, j, grad_loss, fwd_imp))
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]

            grad_lst.append(grad_loss)
            fwd_imp_lst.append(fwd_imp)
        mean_grad_loss = np.mean(grad_lst)
        mean_fwd_imp = np.mean(fwd_imp_lst)
        list_for_sample = []
        for num in range(_num_grad):
            layer_index, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
            if grad_lst[num] >= mean_grad_loss and fwd_imp_lst[num] >= mean_fwd_imp:
                list_for_sample.append([layer_index, i, j, grad_loss, grad_dw0, grad_dw1])
        print(len(list_for_sample))
        return list_for_sample, list_pool, pool

    def _compute_forward_impact_single(self, data_writer_fairness, model, input_data_, candidates_, list_candidate,
                                       data_candidate_dict, num_grad, is_sampled=False, desc=True, based_on=0):
        pool = {}
        list_pool = []
        # candidates, candidates1, candidates2 = candidates_
        layer_index = candidates_[0][0]
        _num_grad = num_grad if num_grad < len(candidates_) else len(candidates_)
        if is_sampled:
            input_data = input_data_  # data_util.sample_negative(input_data_)
        else:
            input_data, _, self.alpha_, self.alpha2, self.alpha3 = data_util.sample_negative(input_data_)
        input_data = input_data_
        input_0, input_1 = data_util.select_subgroups_from_input(input_data)
        fwd_imp_0_list = self._forward_impact(model, input_0, layer_index, list_candidate, _num_grad)
        fwd_imp_1_list = self._forward_impact(model, input_1, layer_index, list_candidate, _num_grad)
        list_pool_sorted = []
        pl_data = {}
        # pl_1 = {}
        for num in range(len(list_candidate)):
            layer_index, i, j, grad_loss, _, _ = list_candidate[num]
            if not i in pl_data.keys():
                pl_data[i] = {}
            pl_data[i][j] = (fwd_imp_0_list[num], fwd_imp_1_list[num])

        list_candidate.sort(key=lambda tup: tup[3], reverse=desc)
        grad_lst = []
        fwd_imp_lst = []
        for num in trange(_num_grad, desc="Computing forward impact"):
            layer_index, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
            fwd_imp_0, fwd_imp_1 = pl_data[i][j]  # pl_0[num], pl_1[num]
            fwd_imp = self._agregate_based_on_fairness_notion(fwd_imp_0, fwd_imp_1, based_on)
            list_pool_sorted.append([layer_index, i, j, grad_loss, fwd_imp])
            list_pool.append((layer_index, i, j, grad_loss, fwd_imp))
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]

            grad_lst.append(grad_loss)
            fwd_imp_lst.append(fwd_imp)

            data_writer_fairness.writerow(
                [num, layer_index, i, j, grad_dw0, grad_dw1, fwd_imp_0, fwd_imp_1, grad_loss, fwd_imp])
        mean_grad_loss = np.mean(grad_lst)
        mean_fwd_imp = np.mean(fwd_imp_lst)
        list_for_sample = []
        for num in range(_num_grad):
            layer_index, i, j, grad_loss, grad_dw0, grad_dw1 = list_candidate[num]
            if grad_lst[num] >= mean_grad_loss and fwd_imp_lst[num] >= mean_fwd_imp:
                list_for_sample.append([layer_index, i, j, grad_loss, grad_dw0, grad_dw1])
        print(len(list_for_sample))
        return list_for_sample, list_pool, pool

    def _reshape_target_model(self, model, input_neg, target_layer=None):
        """Re-shape target model for localize.

        :param model:
        :param input_neg:
        """
        if target_layer is None:
            # "only considers the neural weights connected
            # to the final output layer"
            layer_index = len(model.layers) - 1
            # Search the target layer that satisfies as below
            # 1. The layer is DENSE
            # 2. The output shape corresponds to the final prediction.
            # print(model.layers[layer_index].output.shape, input_neg[1].shape)
            # print(input_neg[1].shape[1])

            while (
                    type(model.get_layer(index=layer_index)) is not tf.keras.layers.Dense
                    or model.layers[layer_index].output.shape[1] != input_neg[1].shape[1]
            ) and layer_index > 0:
                layer_index -= 1
            # update the target_layer
            # because this value is used in _modify_layer_before_reshaped
            self.target_layer = layer_index
        else:
            # Considers the neural weights in designated layer.
            layer_index = self.target_layer
            if layer_index == len(model.layers) - 1:
                # return model
                raise TypeError("Designated layer index is output layer")
        if type(model.get_layer(index=layer_index)) is not tf.keras.layers.Dense:
            raise IndexError(
                "Invalid layer_index: "
                + str(layer_index)
                + " should be keras.layers.core.Dense, but "
                + str(type(model.get_layer(index=layer_index)))
            )
        if layer_index == len(model.layers) - 1:
            return model
        reshaped = tf.keras.models.Model(model.layers[0].input, model.layers[layer_index].output)
        reshaped.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return reshaped

    def _copy_location_to_weights(self, weight, model):
        # for w in location:
        # Parse location data
        val = weight[0]
        layer_index = int(np.round(weight[1]))
        nw_i = int(np.round(weight[2]))
        nw_j = int(np.round(weight[3]))

        # Set neural weight at given position with given value
        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        weights[0][nw_i][nw_j] = val
        layer.set_weights(weights)

        return model

    def _copy_weights_to_location(self, model, weight):
        # for w in location:
        # Parse location data
        layer_index = int(np.round(weight[1]))
        nw_i = int(np.round(weight[2]))
        nw_j = int(np.round(weight[3]))
        # Set weight value in location with neural weight
        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        weight[0] = weights[0][nw_i][nw_j]

        return weight

    def _reshape_labels(self, labels):
        if len(labels.shape) > 1:
            if labels.shape[1] != 1:
                labels = np.argmax(labels, axis=1)
        return labels

    def _get_initial_particle_positions(self, weights, model, num_particles=100):
        # locations = [[0 for j in range(len(weights))] for i in range(self.num_particles)]
        # for n_w in trange(len(weights), desc="Initializing particles"):
        locations = []
        weight = weights[0]
        layer_index = int(weight[0])
        nw_i = int(weight[1])
        nw_j = int(weight[2])

        # "By sibling weights, we refer to all weights
        # corresponding to connections between ... L_{n} and L_{n-1}"
        sibling_weights = []
        # L_{n}
        layer = model.get_layer(index=layer_index)
        target_weights = layer.get_weights()[0]
        for j in range(target_weights.shape[1]):
            for i in range(target_weights.shape[0]):
                # TODO ignore all candidates?
                if j is not nw_j and i is not nw_i:
                    sibling_weights.append(target_weights[i][j])
        # Each element of a particle vector
        # is sampled from a normal distribution
        # defined by the mean and the standard deviation
        # of all sibling neural weighs.
        mu = np.mean(sibling_weights)
        std = np.std(sibling_weights)
        samples = np.random.default_rng().normal(loc=mu, scale=std, size=num_particles)

        for n_p in range(num_particles):
            sample = samples[n_p]
            locations.append([sample, layer_index, nw_i, nw_j])

        return locations

    def update_weights(self, model, location, XA):
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        ### todo check for the fairness here
        test_images, test_labels = XA[0], XA[1]  # , XA[2], XA[3]
        y_pred = model.predict(test_images, verbose=0)

        y_true = test_labels.copy()
        y_pred = self._reshape_labels(y_pred)
        y_true = self._reshape_labels(y_true)

        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)

def _reshape_labels(labels):
    if len(labels.shape) > 1:
        if labels.shape[1] != 1:
            labels = np.argmax(labels, axis=1)
    return labels
def _copy_location_to_weights(location, model):
    for weight in location:
        # Parse location data
        val = weight[0]
        layer_index = int(np.round(weight[1]))
        nw_i = int(np.round(weight[2]))
        nw_j = int(np.round(weight[3]))

        # Set neural weight at given position with given value
        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        weights[0][nw_i][nw_j] = val
        layer.set_weights(weights)

    return model

def _copy_weights_to_location(model, location):
    for weight in location:
        # Parse location data
        layer_index = int(np.round(weight[1]))
        nw_i = int(np.round(weight[2]))
        nw_j = int(np.round(weight[3]))
        # Set weight value in location with neural weight
        layer = model.get_layer(index=layer_index)
        weights = layer.get_weights()
        weight[0] = weights[0][nw_i][nw_j]

    return location
def _identify_pareto(scores):
    """Identify pareto.

    cf. https://pythonhealthcare.org/tag/pareto-front/

    :param scores: Each item has two scores
    :return:
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item.
    # This will then be compared with all other items
    for i in trange(population_size, desc="Identifying pareto-front"):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i'
                # (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]
def _save_pareto_front(scores, pareto_front, output_dir: Path, filename: str):
    """Save pareto front in image."""
    x_all = scores[:, 0]
    y_all = scores[:, 1]
    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]

    plt.scatter(x_all, y_all)
    plt.plot(x_pareto, y_pareto, color="r")
    plt.xlabel("Objective A")
    plt.ylabel("Objective B")

    plt.savefig(output_dir / filename)
def save_weights(weights, output_dir: Path, filename=r"weights.csv"):
    with open(output_dir / filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(weights)
        #self.output_files.add(output_dir / "weights.csv")
def _extract_pareto_front(pool, output_dir=None, filename=r"pareto_front.png"):
    # Compute pareto front
    objectives = []
    for key in tqdm(pool, desc="Collecting objectives for pareto-front"):
        weight = pool[key]
        grad_loss = weight[3]
        fwd_imp = weight[4]
        objectives.append([grad_loss, fwd_imp])
    scores = np.array(objectives)
    pareto = _identify_pareto(scores)
    pareto_front = scores[pareto]

    if output_dir is not None:
        _save_pareto_front(scores, pareto_front, output_dir, filename)

    # Find neural weights on pareto front
    results = []
    for key in tqdm(pool, desc="Extracting pareto-front"):
        weight = pool[key]
        layer_index = weight[0]
        neural_weight_i = weight[1]
        neural_weight_j = weight[2]
        grad_loss = weight[3]
        fwd_imp = weight[4]
        for _pareto_front in pareto_front:
            _grad_loss = _pareto_front[0]
            _fwd_imp = _pareto_front[1]
            if grad_loss == _grad_loss and fwd_imp == _fwd_imp:
                results.append([layer_index, neural_weight_i, neural_weight_j])
                break

    return results


#from src.dnn_testing.eAI.utils.fault_localization_v2 import FaultLocalization
#from src.dnn_testing.eAI.utils.validate_utils import Localization


def _localize(model_dir, input_neg, input_pos, input_both, output_dir: Path, verbose=1, unfav=0, sample_global=False,
              based_on=2, size=2):
    """Localize faulty neural weights.

    NOTE: The arguments of 'weights' and 'loss_func'
          are included in 'model'.

    Parameters
    ----------
    model : repair.core.model.RepairModel
        DNN model to be repaired
    input_neg : tuple[np.ndarray, np.ndarray]
        A set of inputs that reveal the fault
    output_dir : Path, default=Path("outputs")
        Path to directory to save the result
    verbose : int, default=1
        Log level

    Returns
    -------
    A set of neural weights to target for repair

    """

    data_file = open(output_dir / "results-grad-loss_fwd-impact.csv", mode='w', newline='',
                     encoding='utf-8')
    data_writer_fairness = csv.writer(data_file)
    data_writer_fairness.writerow(
        ['pool', 'Layer_index', 'i', 'j', 'grad_loss_0', 'grad_loss_1', 'fwd_impact_0', 'fwd_impact_1',
         'aggre_grad_loss', 'aggre_fwd_impact'])

    model = data_util.load_model_from_tf(Path(model_dir))
    num_grad = len(input_neg[0]) * 20
    target_layer = len(model.layers) - 1
    # print(input_neg[0].shape, num_grad, target_layer)

    # "N_g is set to be the number of negative inputs to repair
    # multiplied by 20"
    # if self.num_grad is None:

    ## todo: modified by moses, to avoid processing output layer
    if target_layer == len(model.layers) - 1:
        target_layer = None
    ## todo: end of updates by moses
    fault_localization = Localization(model_dir, input_neg, input_pos, input_both, unfav=unfav,
                                      sample_global=sample_global)

    reshaped_model = fault_localization._reshape_target_model(model, input_neg)
    # if based_on == 0:
    list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_neg,
                                                                                            based_on=based_on)
    list_for_sample, list_pool, pool = fault_localization._compute_forward_impact_single(data_writer_fairness,
                                                                                         reshaped_model, input_neg,
                                                                                         candidates, list_candidates,
                                                                                         data_candidate_dict, num_grad,
                                                                                         based_on=based_on)
    # weights_t = self._extract_pareto_front(pool, output_dir)
    # weights_t = self._modify_layer_before_reshaped(model, weights_t)
    # #print('1: ', len(pool), len(candidates), len(weights_t))
    list_for_sample = np.array(list_for_sample)

    random_index = np.random.choice(
        [i for i in range(len(list_for_sample))])  # random.randint(0, len(list_for_sample) - 1)
   # random_element = [] list_for_sample[random_index]

    sample = random.choice([i for i in range(len(list_for_sample))], size=size)
    random_element = [list_for_sample[a] for a in sample]

    # print('random_element: ', random_element)
    #
    # # Output neural weight candidates to repair
    # self._append_weights(model, weights_t)
    # self.save_weights(weights_t, output_dir)
    # #print('2: ', len(pool), len(candidates), len(weights_t))
    # self._log_localize(weights_t, verbose)
    # return weights_t
    data_file.close()

    return list_for_sample, random_element, pool

def _modify_layer_before_reshaped(orig_model, weights_t, target_layer=21):
    """Modify the target layer to repair the original target model.

    :param orig_model:
    :param weights_t:
    """

    if target_layer < 0:
        target_layer = len(orig_model.layers) + target_layer
    else:
        target_layer = target_layer
    for weight in weights_t:
        weight[0] = target_layer
    return weights_t
def _append_weights(model, weights):
    for weight in weights:
        layer = model.get_layer(index=int(weight[0]))
        all_weights = layer.get_weights()[0]
        weight.append(all_weights[int(weight[1])][int(weight[2])])
def _weight_value_range(model, layer_index):
    layer = model.get_layer(index=layer_index)
    # Get all weights
    all_weights = []
    target_weights = layer.get_weights()[0]
    for j in range(target_weights.shape[1]):
        for i in range(target_weights.shape[0]):
            all_weights.append(target_weights[i][j])
    #all_weights = [abs(i) for i in all_weights]
    # Velocity bounds defined at equations 5 and 6
    wb = np.min(all_weights) - np.min(all_weights)
    vb = (wb / 5, wb * 5)
    return (np.min(all_weights), np.max(all_weights)), all_weights #, vb
def _get_all_weights(model, layer_index, input_neg, input_pos, input_both, unfav=0, gradient_loss='zero'):
    num_grad = len(input_neg[0]) * 20
    fault_localization = FaultLocalization(model, input_neg, input_pos, input_both, unfav=unfav,
                                           sample_global=False)
    reshaped_model = fault_localization._reshape_target_model(model, input_neg)

    list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_both, based_on=2)
    _, pools = fault_localization._compute_forward_impact_single(reshaped_model, input_both, candidates, list_candidates, data_candidate_dict, num_grad, based_on=2)
    # candidates = fault_localization._compute_gradient_arachne_v1(reshaped_model, input_neg)
    # pools = fault_localization._compute_forward_impact_arachne_v1(reshaped_model, input_neg,candidates, num_grad)
    candidates_list = []
    for num, candidates_ in pools.items():
        candidates_list.append((candidates_[0], candidates_[1], candidates_[2], candidates_[3], candidates_[4]))

    if gradient_loss=='zero':
        candidates_list = [candidate for candidate in candidates_list if candidate[3] > 0]
        candidates_list = [candidate for candidate in candidates_list if candidate[4] > 0]
        location_weights = [[candidate[0], candidate[1], candidate[2]] for candidate in candidates_list]
    if gradient_loss == 'mean':
        candidates_list = [candidate for candidate in candidates_list if candidate[3] > 0]
        candidates_list = [candidate for candidate in candidates_list if candidate[4] > 0]
        grad_list = [candidate[3] for candidate in candidates_list if candidate[3] > 0]
        mean_v = np.mean(grad_list)
        print('mean_v', mean_v, len(candidates_list))

        candidates_list = [candidate for candidate in candidates_list if candidate[3] >= mean_v]
        location_weights = [[candidate[0], candidate[1], candidate[2]] for candidate in candidates_list]
    else:
        location_weights = [[candidate[0], candidate[1], candidate[2]] for candidate in candidates_list]

    print(len(location_weights))
    #
    # layer = model.get_layer(index=layer_index)
    # # Get all weights
    # location_weights = []
    # target_weights = layer.get_weights()[0]
    # print(target_weights.shape)
    # for j in range(target_weights.shape[1]):
    #     for i in range(target_weights.shape[0]):
    #         location_weights.append([layer_index, i, j])
    return location_weights
def run_localize(model_dir, location, input_neg, input_pos, input_both, unfav, based_on, num_grad, num, output_dir, mutated_model=True, avg_neg=None, avg_pos=None, avg_cost=None):
    model = data_util.load_model_from_tf(model_dir)

    if mutated_model:
        orig_location = np.copy(location)
        orig_location = _copy_weights_to_location(model, orig_location)
        model = _copy_location_to_weights(location, model)


    fault_localization = FaultLocalization(model, input_neg, input_pos, input_both, unfav=unfav,
                                           sample_global=False)

    reshaped_model = fault_localization._reshape_target_model(model, input_neg)
    if based_on == 0:
        list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model,
                                                                                                input_neg, avg_cost=avg_neg,
                                                                                                based_on=based_on)
        _, pool = fault_localization._compute_forward_impact_single(reshaped_model, input_neg, candidates,
                                                                    list_candidates, data_candidate_dict, num_grad, avg_cost=avg_neg,
                                                                    based_on=based_on)

    elif based_on == 1:
        list_imp_neg, candidates_neg, data_candidate_dict_neg = fault_localization._compute_gradient(reshaped_model,
                                                                                                     input_neg, based_on=0, avg_cost=avg_neg)
        list_imp_pos, candidates_pos, data_candidate_dict_pos = fault_localization._compute_gradient(reshaped_model,
                                                                                                     input_pos, avg_cost=avg_pos,
                                                                                                     based_on=based_on)
        list_pool, pool = fault_localization._compute_forward_impact_neg_pos(reshaped_model,(input_neg, input_pos),
                                                                             (candidates_neg, candidates_pos),
                                                                             (list_imp_neg, list_imp_pos), (
                                                                             data_candidate_dict_neg,
                                                                             data_candidate_dict_pos), num_grad, avg_cost=(avg_neg, avg_pos, avg_cost),
                                                                             based_on=based_on)
    elif based_on == 2:
        list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model,
                                                                                                input_both, avg_cost=avg_cost,
                                                                                                based_on=based_on)
        _, pool = fault_localization._compute_forward_impact_single(reshaped_model, input_both, candidates,
                                                                    list_candidates, data_candidate_dict, num_grad, avg_cost=avg_cost,
                                                                    based_on=based_on)
    elif based_on == 3:
        list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient_arachne_v2(
            reshaped_model,
            based_on=based_on)
        _, pool = fault_localization._compute_forward_impact_arachne_v2(reshaped_model, candidates,
                                                                        list_candidates, data_candidate_dict,
                                                                        num_grad, based_on=based_on)
    elif based_on == -1:
        candidates = fault_localization._compute_gradient_arachne_v1(reshaped_model, input_neg)
        pool = fault_localization._compute_forward_impact_arachne_v1(reshaped_model, input_neg, candidates, num_grad)

    weights_tg = _extract_pareto_front(pool, output_dir, filename=r"pareto_front-{}-{}.png".format(based_on, num))
    weights_tg = _modify_layer_before_reshaped(model, weights_tg)

    flag = 0
    if mutated_model:
        for w in weights_tg:
            for locate in location:
                if w[0] == locate[1] and locate[2] == w[1] and w[2] == locate[3]:
                    flag += 1
            if flag >=len(location):
                break

    # _append_weights(model, weights_t)
    save_weights(weights_tg, output_dir, filename=r"weights-{}-{}.csv".format(based_on, num))
    if mutated_model:
        model = _copy_location_to_weights(orig_location, model)

    return weights_tg, flag

def _check_model_bias(model, test_images, test_labels, org_pred, fval_org, measure):
    y_pred = model.predict(test_images, verbose=0)
    y_true = test_labels.copy()
    y_pred = _reshape_labels(y_pred)
    y_true = _reshape_labels(y_true)


    Changed_pred = (y_pred != org_pred).sum()
    Changed_pred = Changed_pred * 100 / len(org_pred)

    _, EOD_org, SPD_org, DI_org = fval_org
    fval = _fairness_sub(y_true, y_pred, y_true, measure=measure)
    _, EOD, SPD, DI = fval
    isgreater = False
    if DI >= DI_org and EOD >= EOD_org and SPD >= SPD_org:
        isgreater = True
    # if measure == 'DI' and DI >= DI_org:
    #         isgreater = True
    # elif measure == 'SPD' and SPD >= SPD_org:
    #         isgreater = True
    # elif measure == 'EOD' and EOD >= EOD_org:
    #         isgreater = True
    # else:
    #     if EOD>= EOD_org or SPD >= SPD_org or DI >= DI_org:
    #         isgreater = True
    return isgreater, fval, Changed_pred

def _verify_bias_term(fval_org, fval_new, measure='DI'):
    _, EOD_org, SPD_org, DI_org = fval_org
    _, EOD, SPD, DI = fval_new
    isgreater = False
    if DI >= DI_org and EOD >= EOD_org and SPD >= SPD_org:
        isgreater = True
    # if measure == 'DI' and DI >= DI_org:
    #     isgreater = True
    # elif measure == 'SPD' and SPD >= SPD_org:
    #     isgreater = True
    # elif measure == 'EOD' and EOD >= EOD_org:
    #     isgreater = True
    # else:
    #     #if EOD >= EOD_org or SPD >= SPD_org or DI >= DI_org:
    #     isgreater = False
    return isgreater
def _compute_class_weights(y_true, y_pred):
    sensitive_unique = np.unique(y_true)
    sensitive_unique.sort()

    Count_0, Count_1 = 0, 0
    P_obs_0, P_obs_1 = 0, 0

    unpriveledge = sensitive_unique[0]

    ### F=0, M=1
    N = len(y_true)
    M = (y_true == 1).sum()
    F = (y_true == 0).sum()

    MC = ((y_true == y_pred) & (y_true == 1)).sum()
    FC = ((y_true == y_pred) & (y_pred == 0)).sum()
    MM = ((y_true != y_pred) & (y_pred == 0)).sum()
    FM = ((y_true != y_pred) & (y_pred == 1)).sum()

    W_MC = ((M/N)*(MC/N)) / (MC/N)
    W_MM = (M / N) * (MM / N) / (MM / N)
    W_FC = (F / N) * (FC / N) / (FC / N)
    W_FM = (F / N) * (FM / N) / (FM / N)

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
    W_MM = (M / N) * (MM / N) / (M_ / N)
    W_FC = (F / N) * (FC / N) / (C / N)
    W_FM = (F / N) * (FM / N) / (M_ / N)

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
def _value_bound(min_val, max_val, val):
    new_val = val
    if val >= min_val and val <= max_val:
        return new_val
    else:
        if val > max_val:
            new_val = max_val
        elif val < min_val:
            new_val = min_val
        return new_val


def _parse_results(test_images, test_labels, pred_labels):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    input_neg = (test_images[neg_indices], test_labels[neg_indices],
                 np.argmax(test_labels[neg_indices], axis=1).reshape(-1, 1))
    input_pos = (test_images[pos_indices], test_labels[pos_indices],
                 np.argmax(test_labels[pos_indices], axis=1).reshape(-1, 1))

    input_both = (np.vstack((input_neg[0], input_pos[0])),
                  np.vstack((input_neg[1], input_pos[1])),
                  np.vstack((input_neg[2], input_pos[2])))
    return input_neg, input_pos, input_both


def _parse_results_uniform_sampling_positive_same(test_images, test_labels, pred_labels, size_mc, size_mm, size_fc, size_fm):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    #pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    MC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 1))[0]
    FC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 0))[0]

    MM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 1))[0]
    FM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 0))[0]

    pos_size = len(MC_Indices)
    if MC_Indices > FC_Indices:
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

def _parse_results_uniform_sampling_positive(test_images, test_labels, pred_labels, size_mc, size_mm, size_fc, size_fm):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    #pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

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

def _parse_results_uniform_sampling(test_images, test_labels, pred_labels, size_mc, size_mm, size_fc, size_fm):
    y_true_argmax = np.argmax(test_labels, axis=1)
    y_pred_argmax = np.argmax(pred_labels, axis=1)
    #neg_indices = np.where(y_true_argmax != y_pred_argmax)[0]
    #pos_indices = np.where(y_true_argmax == y_pred_argmax)[0]

    MC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 1))[0]
    FC_Indices = np.where((y_true_argmax == y_pred_argmax) & (y_true_argmax == 0))[0]

    MM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 1))[0]
    FM_Indices = np.where((y_true_argmax != y_pred_argmax) & (y_true_argmax == 0))[0]

    sample_MC = np.random.choice(MC_Indices, size=size_mc)
    sample_FC = np.random.choice(FC_Indices, size=size_fc)
    sample_MM = np.random.choice(MM_Indices, size=size_mm)
    sample_FM = np.random.choice(FM_Indices, size=size_fm)

    input_neg = (np.vstack((test_images[sample_FM], test_images[sample_MM])),
                 np.vstack((test_labels[sample_FM], test_labels[sample_MM])),
                 np.vstack((np.argmax(test_labels[sample_FM], axis=1).reshape(-1,1),
                            np.argmax(test_labels[sample_MM], axis=1).reshape(-1,1))))

    input_pos = (np.vstack((test_images[sample_FC], test_images[sample_MC])),
                 np.vstack((test_labels[sample_FC], test_labels[sample_MC])),
                 np.vstack((np.argmax(test_labels[sample_FC], axis=1).reshape(-1, 1),
                            np.argmax(test_labels[sample_MC], axis=1).reshape(-1, 1))))

    input_both = (np.vstack((input_neg[0], input_pos[0])), np.vstack((input_neg[1], input_pos[1])))

    return input_neg, input_pos, input_both


def localize_original_model(model_dir, model, input_neg, input_pos, input_both, org_pred, output_dir, unfav, num, fval_org, measure='DI', verbose=0, based_on=0, cost_sensitive=True):
    test_images, test_labels = input_both[0], input_both[1]  # , XA[2], XA[3]
    y_pred = model.predict(test_images, verbose=0)
    y_true = test_labels.copy()
    y_pred = _reshape_labels(y_pred)
    y_true = _reshape_labels(y_true)

    isgreater, fval, Changed_pred = _check_model_bias(model, test_images, test_labels, org_pred, fval_org,
                                                      measure=measure)

    # if self.num_grad is None:
    num_grad = len(input_neg[0]) * 20
    ## todo: modified by moses, to avoid processing output layer
    # target_layer == len(model.layers) - 1:
    target_layer = None

    if cost_sensitive:
        _, avg_cv_prime_neg, avg_cost_neg = compute_predict_prima(model, input_neg)
        _, avg_cv_prime_pos, avg_cost_pos = compute_predict_prima(model, input_pos)
        _, avg_cv_prime, avg_cost = compute_predict_prima(model, input_both)
    else:
        avg_cv_prime_neg, avg_cost_neg = None, None
        avg_cv_prime_pos, avg_cost_pos = None, None
        avg_cv_prime, avg_cos = None, None

    ## todo: end of updates by moses
    unfav, W_M, W_F, W_MC, W_MM, W_FC, W_FM, MC, MM, FC, FM = _get_subgroup_names_as_binary(y_true, y_pred)
    based_on = 0
    weights_tg, flag_neg = run_localize(model_dir, None, input_neg, input_pos, input_both, unfav,
                                        based_on, num_grad, num, output_dir, mutated_model=False, avg_neg=avg_cv_prime_neg, avg_pos=avg_cv_prime_pos, avg_cost=avg_cv_prime)
    based_on = -1
    weights_arachne1, flag_arachne1 = run_localize(model_dir, None, input_neg, input_pos, input_both,
                                                   unfav,
                                                   based_on, num_grad, num, output_dir, mutated_model=False, avg_neg=avg_cv_prime_neg, avg_pos=avg_cv_prime_pos, avg_cost=avg_cv_prime)
    based_on = 1
    weights_both, flag_both = run_localize(model_dir, None, input_neg, input_pos, input_both,
                                           unfav,
                                           based_on, num_grad, num, output_dir, mutated_model=False, avg_neg=avg_cv_prime_neg, avg_pos=avg_cv_prime_pos, avg_cost=avg_cv_prime)

    based_on = 2
    weights_mixed, flag_mixed = run_localize(model_dir, None, input_neg, input_pos, input_both,
                                             unfav,
                                             based_on, num_grad, num, output_dir, mutated_model=False, avg_neg=avg_cv_prime_neg, avg_pos=avg_cv_prime_pos, avg_cost=avg_cv_prime)
    based_on = 3  ## Arachne v2,
    weights_arachne2, flag_arachne2 = run_localize(model_dir, None, input_neg, input_pos, input_both,
                                                   unfav,
                                                   based_on, num_grad, num, output_dir, mutated_model=False, avg_neg=avg_cv_prime_neg, avg_pos=avg_cv_prime_pos, avg_cost=avg_cv_prime)

    return fval, 0, Changed_pred, flag_neg, flag_both, flag_mixed, flag_arachne1, flag_arachne2, weights_tg, weights_both, weights_mixed, weights_arachne1, weights_arachne2
def update_weights(model_dir, model, locations, weights_bounds, input_neg_, input_pos_, input_both_, org_pred, output_dir, unfav, num, fval_org, measure='DI', verbose=0, based_on=0, cost_sensitive=True):
    location_org = []
    global_new_val = []
    for locate in locations:
        location_org.append([locate[1], locate[2], locate[3]])
        global_new_val.append(locate[0])
    orig_location = np.copy(locations)
    orig_location = _copy_weights_to_location(model, orig_location)
    model = _copy_location_to_weights(locations, model)
    # arachne._append_weights(model, weights)
    #arachne._output_repaired_model(output_dir, model)

    ### todo check for the fairness here

    test_images, test_labels = input_both_[0], input_both_[1]  # , XA[2], XA[3]
    isgreater, fval, Changed_pred = _check_model_bias(model, test_images, test_labels, org_pred, fval_org, measure=measure)
    # y_pred = model.predict(test_images, verbose=0)
    #
    # y_true = test_labels.copy()
    # y_pred = _reshape_labels(y_pred)
    # y_true = _reshape_labels(y_true)
    #
    # Changed_pred = (y_pred != org_pred).sum()
    # Changed_pred = Changed_pred * 100 / len(org_pred)
    #
    # f_val = _fairness_sub(y_true, y_pred, y_true, measure='AOD')
    #
    # _, EOD_org, SPD_org, DI_org = fval_org
    # _, EOD, SPD, DI = f_val
    #value_range = (0, 1)
    bounds, all_weights = weights_bounds
    if isgreater:
        pass
    else:

        for i in range(10):
            #location_ = []
            for locate_index in range(len(locations)):
                locate = locations[locate_index]
                new_val = locate[0]
                #sample = random.uniform(bounds[0], bounds[1])


                #noise = np.random.normal(loc=0, scale=1, size=100)
                sample = random.choice(all_weights, size=3)
                #sample = [random.uniform(bounds[0], bounds[1]), new_val]
                #sample.append(new_val)

                #choice_ = np.sum([random.uniform(bounds[0], bounds[1]), locate[0]])
                #sample = [abs(a) for a in sample]
                choice_ = np.sum([a for a in sample])
                # choice_ = np.random.choice(noise)
                # if locate[0] < 0:
                #     new_val = 0-choice_
                # else:
                #     new_val = choice_
                new_val = choice_
                #new_val = _value_bound(bounds[0], bounds[1], new_val)
                #global_new_val += abs(new_val)
                global_new_val[locate_index] = new_val
                print(new_val, ' ---- loooping at ' + str(i + 1) + ' ----', fval)
                #if abs(new_val) > abs(va)
                locations[locate_index] = (new_val, locate[1], locate[2], locate[3])
            model = _copy_location_to_weights(locations, model)
            isgreater, fval, Changed_pred = _check_model_bias(model, test_images, test_labels, org_pred, fval_org,
                                                              measure=measure)
            if isgreater:
                break
    pred = model.predict(test_images, verbose=0)
    y_true = test_labels.copy()
    y_pred = _reshape_labels(pred)
    y_true = _reshape_labels(y_true)

    print('checking the shape, y_true: ', y_true.shape, 'y_pred: ', y_pred.shape)
    if isgreater == False:
        # Restore original weights to the model
        model = _copy_location_to_weights(orig_location, model)
        return None

    # if cost_sensitive:
    #     _, avg_cv_prime_neg, avg_cost_neg = compute_predict_prima(model, input_neg)
    #     _, avg_cv_prime_pos, avg_cost_pos = compute_predict_prima(model, input_pos)
    #     _, avg_cv_prime, avg_cost = compute_predict_prima(model, input_both)
    # else:
    #     avg_cv_prime_neg, avg_cost_neg = None, None
    #     avg_cv_prime_pos, avg_cost_pos = None, None
    #     avg_cv_prime, avg_cos = None, None

    # print(f_val)
    #if self.num_grad is None:

    ## todo: modified by moses, to avoid processing output layer
    #target_layer == len(model.layers) - 1:
    target_layer = None
    ## todo: end of updates by moses
    model = _copy_location_to_weights(orig_location, model)



    unfav, W_M, W_F, W_MC, W_MM, W_FC, W_FM, MC, MM, FC, FM = _get_subgroup_names_as_binary(y_true, y_pred)
    avg_neg = (W_FM/(W_FM+W_MM), W_MM/(W_FM+W_MM))
    avg_pos = (W_FC/(W_FC+W_MC), W_MC/(W_FC+W_MC))
    avg_both = (W_F/(W_F+W_M), W_M/(W_F+W_M))
    #nput_neg, input_pos, input_both = _parse_results_uniform_sampling(test_images,test_labels, pred, int(W_MC*MC), int(W_MM*MM), int(W_FC*FC), int(W_FM*FM))
    input_neg, input_pos, input_both = _parse_results(test_images, test_labels, pred)
    num_grad = len(input_neg[0]) * 20

    based_on = 0
    weights_tg, flag_neg = run_localize(model_dir, locations, input_neg, input_pos, input_both,unfav,based_on,num_grad,num,output_dir, avg_neg=avg_neg, avg_pos=avg_pos, avg_cost=avg_both)
    based_on = -1
    weights_arachne1, flag_arachne1 = run_localize(model_dir, locations, input_neg, input_pos, input_both, unfav,
                                        based_on, num_grad, num, output_dir, avg_neg=avg_neg, avg_pos=avg_pos, avg_cost=avg_both)
    based_on = 1
    weights_both, flag_both = run_localize(model_dir, locations, input_neg, input_pos, input_both,
                                                   unfav,
                                                   based_on, num_grad, num, output_dir, avg_neg=avg_neg, avg_pos=avg_pos, avg_cost=avg_both)

    based_on = 2
    weights_mixed, flag_mixed = run_localize(model_dir, locations, input_neg, input_pos, input_both,
                                           unfav,
                                           based_on, num_grad, num, output_dir, avg_neg=avg_neg, avg_pos=avg_pos, avg_cost=avg_both)
    based_on = 3 ## Arachne v2,
    weights_arachne2, flag_arachne2 = run_localize(model_dir, locations, input_neg, input_pos, input_both,
                                             unfav,
                                             based_on, num_grad, num, output_dir, avg_neg=avg_neg, avg_pos=avg_pos, avg_cost=avg_both)

    # Restore original weights to the model
    model = _copy_location_to_weights(orig_location, model)

    return fval, global_new_val, Changed_pred, flag_neg, flag_both, flag_mixed, flag_arachne1, flag_arachne2, weights_tg, weights_both, weights_mixed, weights_arachne1, weights_arachne2
def update_model_weight(model, layer_index, i, j):
    noise = np.random.normal(loc=0, scale=1, size=10)
    #print(noise)
    layer = model.get_layer(index=layer_index)
    weights = layer.get_weights()
    val_org = weights[0][i][j]
    choice_ = np.random.choice(noise)
    new_val = val_org
    if val_org < 0:
        new_val -= abs(choice_)
    else:
        new_val += abs(choice_)
    location = (new_val, layer_index, i, j)
    #print(val_org, new_val, choice_)
    return location, val_org, new_val, abs(choice_)


def _compute_proba_pred(true, pred):
    true_argmax = np.argmax(true, axis=1)
    pred_argmax = np.argmax(pred, axis=1)
    prob_pred = []
    costs_0 = 1
    costs_1 = 1

    for i in range(pred_argmax.shape[0]):
        if true_argmax[i] == 0:
            if pred_argmax[i] != true_argmax[i]:
                costs_0 += 1
        else:
            if pred_argmax[i] != true_argmax[i]:
                costs_1 += 1
        cl_1 = pred[:, 1][i] / np.sum(pred[:, 0])
        cl_0 = pred[:, 0][i] / np.sum(pred[:, 1])
        prob_pred.append([cl_0, cl_1])

    # print()
    prob_pred = np.array(prob_pred)
    return costs_0, costs_1, prob_pred


def compute_cost_vector(true, pred):
    costs_0, costs_1, prob_pred = _compute_proba_pred(true, pred)
    print("costs_0", costs_0, "costs_1", costs_1)
    # print("prob_pred", prob_pred)
    true_argmax = np.argmax(true, axis=1)
    pred_argmax = np.argmax(pred, axis=1)
    N = true.shape[0]
    sum_cost_prob = 1
    for i in range(pred_argmax.shape[0]):
        if true_argmax[i] != pred_argmax[i]:
            if true_argmax[i] == 1:
                sum_cost_prob += prob_pred[:, 1][i] * costs_0
            else:
                sum_cost_prob += prob_pred[:, 0][i] * costs_1
    cost_vector = []
    for i in range(pred_argmax.shape[0]):
        cst_vec_1 = (1 / (1 - prob_pred[:, 1][i])) * sum_cost_prob
        cst_vec_0 = (1 / (1 - prob_pred[:, 0][i])) * sum_cost_prob
        cost_vector.append([cst_vec_0, cst_vec_1])
    cost_vector = np.array(cost_vector)
    return cost_vector, prob_pred, (costs_0 / N, costs_1 / N)


def compute_predict_prima(model, input_data):
    # true = true.numpy()
    x, y = input_data[0], input_data[1]
    x = K.constant(x)
    pred = model(x)
    pred = pred.numpy()
    ##todo: equation 12
    cost_vector, proba, avg_cost = compute_cost_vector(y, pred)
    avg_cost_vector = [np.mean(cost_vector[:, 0]), np.mean(cost_vector[:, 1])]
    cv_prime = np.array(
        [[cost_vector[:, 0][i] / max(cost_vector[:, 0]), cost_vector[:, 1][i] / max(cost_vector[:, 1])] for i in
         range(len(pred))])
    avg_cv_prime = [np.mean(cv_prime[:, 0]), np.mean(cv_prime[:, 1])]
    # pred_prime = [[v[0]*cost_vector[:,0][i]/max(cost_vector[:,0]),v[1]*cost_vector[:,1][i]/max(cost_vector[:,1])] for i, v in enumerate(pred)]
    return avg_cost_vector, avg_cv_prime, avg_cost