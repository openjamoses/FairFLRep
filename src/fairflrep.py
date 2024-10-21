import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.utils.fairness import calculate_Disparate_Impact, calculate_SPD, \
    calculate_equal_opportunity_difference, calculate_average_odds_difference, calculate_FPR_difference, \
    calculate_TPR_difference
import tensorflow as tf
from tensorflow.keras import backend as K  # noqa: N812
from pathlib import Path
from tqdm import tqdm, trange
from src.dnn_testing.eAI.utils import data_util
import shutil
from src.dnn_testing.eAI.utils import fairness
import random
import logging
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils.fault_localization_mixed import FaultLocalization

logger = logging.getLogger(__name__)
logger.disabled = True

import shutil
class EvaluateFairness:
    def __init__(self,
                 model_dir,
                 target_data_dir: Path,
                 data_dict,
                 age_test_label,
                 positive_inputs_dir: Path,
                 output_dir: Path,):
        self.model_dir = model_dir
        self.target_data_dir = target_data_dir
        self.data_dict = data_dict
        self.age_test_label = age_test_label
        self.positive_inputs_dir = positive_inputs_dir
        self.output_dir = output_dir
    def evaluate(self,
                 num_runs,
                 fairness_measure,
                 unfav,
                 sample_global,
                 based_on,
                 start_iteration=0,
                 topN=10,
                 verbose=1,):

        # Make output directory if not exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []
        # input_neg = self.data_dict['NEG']
        # input_pos = self.data_dict['POS']
        # input_both = self.data_dict['BOTH']

        for i in range(start_iteration, num_runs):
            input_neg = self.data_dict[i]['NEG']
            input_pos = self.data_dict[i]['POS']
            input_both = self.data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(self.model_dir)
            print(i, 'model path: ', self.model_dir)
            # Localize
            localized_data_dir = self.output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            arachne = FairFLRep()
            arachne.localize_bias(self.model_dir, input_neg, input_pos, input_both, localized_data_dir, verbose, unfav=unfav,
                               sample_global=False, based_on=based_on, topN=topN)

            # Optimize
            weights = arachne.load_weights(localized_data_dir_path)
            repaired_model_dir = self.output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()


            arachne.optimize(
                model,
                self.model_dir,
                weights,
                input_neg,
                input_pos,
                input_both,
                self.age_test_label,
                repaired_model_dir,
                fairness_measure,
                i,
                verbose,
            )
    def evaluate_arachne(self,
                 num_runs,
                 fairness_measure,
                 unfav,
                 sample_global,
                 based_on,
                start_iteration=0,
                topN=10,
                 verbose=1, ):

        # Make output directory if not exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        score_rr = []
        score_br = []

        for i in range(num_runs):
            input_neg = self.data_dict[i]['NEG']
            input_pos = self.data_dict[i]['POS']
            input_both = self.data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(self.model_dir)
            print(i, 'model path: ', self.model_dir)
            # Localize
            localized_data_dir = self.output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            arachne = Arachne()
            arachne.localize_bias(self.model_dir, input_neg, input_pos, input_both, localized_data_dir, verbose,
                                  unfav=unfav,
                                  sample_global=False, based_on=based_on,topN=topN)

            # Optimize
            weights = arachne.load_weights(localized_data_dir_path)
            repaired_model_dir = self.output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()

            arachne.optimize(
                model,
                self.model_dir,
                weights,
                input_neg,
                input_pos,
                input_both,
                self.age_test_label,
                repaired_model_dir,
                fairness_measure,
                verbose,
            )
class FairFLRep:
    def __init__(self):
        """Initialize."""
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 100
        self.num_input_pos_sampled = 200
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = None
        self.output_files = set()
        self.batch_size = 32
        # self.random_seed = 0
        # np.random.seed(self.random_seed)
    def localize_bias(self, model_dir, input_neg, input_pos, input_both, output_dir: Path, verbose=1, unfav=0, sample_global=False, based_on=2, topN=10):
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
        print(input_neg[0].shape, self.num_grad, self.target_layer)
        model = data_util.load_model_from_tf(Path(model_dir))
        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20
        ## todo: modified by moses, to avoid processing output layer
        if self.target_layer == len(model.layers) - 1:
            self.target_layer = None
        ## todo: end of updates by moses
        fault_localization = FaultLocalization(model_dir, input_neg, input_pos, input_both, unfav=unfav,
                                               sample_global=sample_global)

        reshaped_model = self._reshape_target_model(model, input_both)

        if based_on == 1:
            list_imp_neg, candidates_neg, data_candidate_dict_neg = fault_localization._compute_gradient(reshaped_model, input_neg,
                                                                                                   based_on=based_on, local_base_on=0)
            list_imp_pos, candidates_pos, data_candidate_dict_pos = fault_localization._compute_gradient(reshaped_model, input_pos,
                                                                                based_on=based_on, local_base_on=1)
            list_pool, pool = fault_localization._compute_forward_impact_neg_pos(reshaped_model, (candidates_neg, candidates_pos), (list_imp_neg, list_imp_pos), (data_candidate_dict_neg, data_candidate_dict_pos), self.num_grad, based_on=3)
        elif based_on == 2: ## todo: Both positive and negative, without knowledge of sensitive
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_both, based_on=based_on, local_base_on=2)
            _, pool = fault_localization._compute_forward_impact_single(reshaped_model, input_both, candidates, list_candidates, data_candidate_dict, self.num_grad, based_on=based_on)
        elif based_on == 3: ## Arachne v2,
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient_arachne_v2(reshaped_model,
                                                                                                    based_on=based_on)
            _, pool = fault_localization._compute_forward_impact_arachne_v2(reshaped_model, candidates,
                                                                        list_candidates, data_candidate_dict,
                                                                        self.num_grad, based_on=based_on)
        elif based_on == 4 or based_on == 6: ## todo: Both positive and negative, without knowledge of sensitive
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_both, based_on=based_on, local_base_on=2)
            _, pool = fault_localization._compute_forward_impact_single(reshaped_model, input_both, candidates, list_candidates, data_candidate_dict, self.num_grad, based_on=based_on)
        elif based_on == 5:
            list_imp_neg, candidates_neg, data_candidate_dict_neg = fault_localization._compute_gradient(reshaped_model,
                                                                                                         input_neg,
                                                                                                         based_on=based_on,
                                                                                                         local_base_on=0)
            list_imp_pos, candidates_pos, data_candidate_dict_pos = fault_localization._compute_gradient(reshaped_model,
                                                                                                         input_pos,
                                                                                                         based_on=based_on,
                                                                                                         local_base_on=1)
            list_pool, pool = fault_localization._compute_forward_impact_neg_pos(reshaped_model,
                                                                                 (candidates_neg, candidates_pos),
                                                                                 (list_imp_neg, list_imp_pos), (
                                                                                 data_candidate_dict_neg,
                                                                                 data_candidate_dict_pos),
                                                                                 self.num_grad, based_on=3)
        elif based_on == -2:
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model,
                                                                                                    input_both,
                                                                                                    based_on=2)
            weights_t = candidates[:topN]
        elif based_on == -3:
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient_arachne_v2(
                reshaped_model,
                based_on=based_on)
            weights_t = candidates[:topN]
        #
        if based_on == 1 or based_on == 2 or based_on == 3 or based_on == 4 or based_on == 5 or based_on == 6:
            weights_t = self._extract_pareto_front(pool, output_dir)
            #print('\n')
            #print('0: ', len(pool), len(candidates), len(weights_t))
            weights_t = self._modify_layer_before_reshaped(model, weights_t)
        #print('1: ', len(pool), len(candidates), len(weights_t))
        # Output neural weight candidates to repair
        self._append_weights(model, weights_t)
        self.save_weights(weights_t, output_dir)
        #print('2: ', len(pool), len(candidates), len(weights_t))
        self._log_localize(weights_t, verbose)
        return weights_t

    def localize(self, model, input_neg, output_dir: Path, verbose=1):
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
        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20
        ## todo: modified by moses, to avoid processing output layer
        if self.target_layer == len(model.layers) - 1:
            self.target_layer = None
        ## todo: end of updates by moses

        reshaped_model = self._reshape_target_model(model, input_neg)

        candidates = self._compute_gradient(reshaped_model, input_neg)

        pool = self._compute_forward_impact(reshaped_model, input_neg, candidates, self.num_grad)

        weights_t = self._extract_pareto_front(pool, output_dir)

        weights_t = self._modify_layer_before_reshaped(model, weights_t)

        # Output neural weight candidates to repair
        self._append_weights(model, weights_t)
        self.save_weights(weights_t, output_dir)

        self._log_localize(weights_t, verbose)
        return weights_t

    def _get_subgroup_names_as_binary(self, sensitive_input):
        # keys = list(fitness_dict.keys())
        sensitive_unique = np.unique(sensitive_input)
        sensitive_unique.sort()
        self.group_a, self.group_b = [], []
        for i in range(int(len(sensitive_unique) / 2)):
            self.group_a.append(sensitive_unique[i])
        for i in range(int(len(sensitive_unique) / 2), len(sensitive_unique)):
            self.group_b.append(sensitive_unique[i])
        return self.group_a, self.group_b

    def optimize(
            self,
            model,
            model_dir: Path,
            weights,
            input_neg,
            input_pos,
            XA,
            sensitive_input,
            output_dir: Path,
            fairness_measure='DI',
            global_epoch=0,
            verbose=1,
    ):
        """Optimize.

        cf. https://qiita.com/sz_dr/items/bccb478965195c5e4097

        Parameters
        ----------
        model :
            DNN model to repair
        model_dir : Path
            (Not used)
        weights
            Set of neural weights to target for repair
        input_neg
            Dataset for unexpected behavior
        input_pos
            Dataset for correct behavior
        output_dir : Path
            Path to directory to save result
        verbose : int, default=1
            Log level

        """
        group_a, group_b = self._get_subgroup_names_as_binary(sensitive_input)
        # Initialize particle positions
        locations = self._get_initial_particle_positions(weights, model)

        # "The initial velocity of each particle is set to zero"
        velocities = np.zeros((self.num_particles, len(weights)))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds(model)

        input_pos_sampled = input_pos
        input_neg_sampled = input_neg
        # "We sample 200 positive inputs"
        #input_pos_sampled, pos_sample = self._sample_positive_inputs(input_pos)

        #input_neg_sampled, neg_sample = self._sample_positive_inputs(input_neg)
        # Convert the dataset into numpy-format not to use generators of tensorflow.
        # When the number of dataset is not so large,
        # the cost of making generators can not be ignored, and it causes memory-leak.
        # neg_sample = input_neg.sample_from_file(input_pos_sampled[0].shape[0])
        # input_neg_sampled = (input_neg[0][neg_sample], input_neg[1][neg_sample], input_neg[2][neg_sample])
        sensitive_input_org = sensitive_input.copy()
        # Initialize for PSO search
        personal_best_positions = list(locations)

        # for n_p in range(self.num_particles):
        #     sample = samples[n_p]
        #     locations[n_p][n_w] = (sample, layer_index, nw_i, nw_j)

        personal_best_scores, _ = self._initialize_personal_best_scores(
            locations, model, input_pos_sampled, input_neg_sampled, XA, sensitive_input_org,
            fairness_measure=fairness_measure
        )
        ## todo: we should update here too!
        # fitness_dict, fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
        ## convert to binary
        # fitness_dict[key] = ((neg_fit + pos_fit), (AOD, EOD, SPD, DI))
        # list_abs_fit = self._abs_fitness_list(personal_best_scores_with_dict, personal_best_scores)
        best_particle = np.argmin(np.array(personal_best_scores)[:, 0])
        # best_particle = np.argmax(np.array(list_abs_fit))
        global_best_position = personal_best_positions[best_particle]

        # Search
        history = []
        performance_history = []
        # "PSO uses ... the maximum number of iterations is 100"
        for t in range(self.num_iterations):
            g = self._get_weight_values(global_best_position)
            # "PSO uses a population size of 100"

            # fitness_dict_list = []
            score_list = []
            n_patched_list = []
            n_intact_list = []
            for n in trange(
                    self.num_particles,
                    desc="Updating particle positions" f" (it={t + 1}/{self.num_iterations})",
            ):
                new_weights, new_v, score, f_metric, p_metric = self._update_particle(
                    locations[n],
                    velocities[n],
                    velocity_bounds,
                    personal_best_positions[n],
                    g,
                    model,
                    input_pos_sampled,
                    input_neg_sampled,
                    XA,
                    sensitive_input_org,
                    fairness_measure
                )

                # fitness_dict_list.append(fitness_dict)
                score_list.append(score)
                #n_patched_list.append(n_patched)
                #n_intact_list.append(n_intact)
                # Update position
                locations[n] = new_weights
                # Update velocity
                velocities[n] = new_v

                # _fairness_sub = self._fairness_sub(fitness_dict, measure=fairness_measure)
                ## Todo: overall_score = score -  fairness constraints
                # list_abs_fit.append(fitness-abs(_fairness_sub))
                # overall_score = _fairness_sub #score-abs(_fairness_sub)

                # Update score
                if personal_best_scores[n][0] > score:
                    # print(personal_best_scores[n][0], score)
                    personal_best_scores[n] = [score, f_metric, p_metric]
                    personal_best_positions[n] = locations[n]

                    performance_history.append([score, f_metric, p_metric])
            # list_abs_fit = self._abs_fitness_list(fitness_dict_list, score_list)
            # Update global best
            # best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
            best_particle = np.argmin(np.array(personal_best_scores)[:, 0])
            # best_particle = np.argmax(np.array(list_abs_fit))
            global_best_position = personal_best_positions[best_particle]
            # Add current best
            history.append(personal_best_scores[best_particle])

            # Stop earlier
            # if self._fail_to_find_better_patch(t, history):
            #     break
            if self._fail_to_find_better_patch_fairness(t, history):
                break
        try:
            self.save_metrics(performance_history, global_epoch, fairness_measure, output_dir)
            self.plot_performance(performance_history, global_epoch, fairness_measure, output_dir)
        except Exception as e:
            print('Error: ', e)
        self._append_weights(model, weights)
        model = self._copy_location_to_weights(global_best_position, model)
        self._append_weights(model, weights)
        self.save_weights(weights, output_dir)

        self._output_repaired_model(output_dir, model)
        self._log_optimize(global_best_position, verbose)

        return model
    def save_metrics(self, history, epoch, fairness, output_dir):
        with open(output_dir / "metrics-{}-{}.csv".format(fairness, epoch), "w") as f:
            writer = csv.writer(f)
            writer.writerows(history)
    def plot_performance(self, history, epoch, fairness, output_dir):
        x = [i for i in range(len(history))]
        # y1 = [1000, 13000, 26000, 42000, 60000, 81000]
        # y2 = [1000, 13000, 27000, 43000, 63000, 85000]
        history = np.array(history)
        # Clear the entire figure
        plt.clf()
        plt.plot(x, history[:, 0], label=fairness)
        plt.plot(x, history[:, 1], '-.', label='Acc')
        #plt.plot(x, history[:, 2], '-.', label='Arachne')

        # plt.xlabel("X-axis data")
        # plt.ylabel("Y-axis data")
        plt.legend()
        # plt.title('multiple plots')
        plt.savefig(output_dir / 'metric-{}-{}.png'.format(fairness, epoch), bbox_inches='tight')
        #plt.show()
        # Close the figure
        plt.close()


    def evaluate_bias(
            self,
            dataset,
            model_dir: Path,
            #target_data,
            target_data_dir: Path,
            data_dict,
            age_test_label,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            fairness_measure,
            unfav,
            sample_global,
            based_on,
            verbose=1,

    ):
        """Evaluate.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing
        verbose : int, default=1
            Log level

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []
        input_neg = data_dict['NEG']
        input_pos = data_dict['POS']
        input_both = data_dict['BOTH']

        for i in range(num_runs):
            # input_neg = data_dict[i]['NEG']
            # input_pos = data_dict[i]['POS']
            # input_both = data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(model_dir)
            print(i, 'model path: ', model_dir)

            # Localize
            localized_data_dir = output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            self.localize_bias(model_dir, input_neg, input_pos, input_both, localized_data_dir, verbose, unfav=unfav, sample_global=False,based_on=based_on)

            # Optimize
            weights = self.load_weights(localized_data_dir_path)
            repaired_model_dir = output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()
            self.optimize(
                model,
                model_dir,
                weights,
                input_neg,
                input_pos,
                input_both,
                age_test_label,
                repaired_model_dir,
                fairness_measure,
                verbose,
            )
            # # Compute RR
            # model = data_util.load_model_from_tf(repaired_model_dir / "repair")
            # repair_target_dataset = data_util.load_repair_data(target_data_dir)
            # repair_images, repair_labels = (
            #     repair_target_dataset[0],
            #     repair_target_dataset[1],
            # )
            #
            # score = model.evaluate(
            #     repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            # )
            # rr = score[1] * 100
            # score_rr.append(rr)
            #
            # # Compute BR
            # repair_positive_dataset = data_util.load_repair_data(positive_inputs_dir)
            # repair_images, repair_labels = (
            #     repair_positive_dataset[0],
            #     repair_positive_dataset[1],
            # )
            #
            # score = model.evaluate(
            #     repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            # )
            # br = (1 - score[1]) * 100
            # score_br.append(br)

        # Output results
        self._save_evaluate_results(
            dataset,
            model_dir,
            target_data_dir,
            positive_inputs_dir,
            output_dir,
            num_runs,
            score_rr,
            score_br,
        )
        self._log_evaluate(score_rr, score_br, num_runs, verbose)

    def evaluate_v2(
            self,
            dataset,
            model_dir: Path,
            #target_data,
            target_data_dir: Path,
            data_dict,
            age_test_label,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            fairness_measure,
            verbose=1,
    ):
        """Evaluate.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing
        verbose : int, default=1
            Log level

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []

        for i in range(num_runs):
            target_data = data_dict[i]['NEG']
            positive_inputs = data_dict[i]['POS']
            XA = data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(model_dir)
            print(i, 'model path: ', model_dir)

            # Localize
            localized_data_dir = output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            self.localize(model, target_data, localized_data_dir, verbose)

            # Optimize
            weights = self.load_weights(localized_data_dir_path)
            repaired_model_dir = output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()
            self.optimize(
                model,
                model_dir,
                weights,
                target_data,
                positive_inputs,
                XA,
                age_test_label,
                repaired_model_dir,
                fairness_measure,
                verbose,
            )
            # Compute RR
            model = data_util.load_model_from_tf(repaired_model_dir / "repair")
            repair_target_dataset = data_util.load_repair_data(target_data_dir)
            repair_images, repair_labels = (
                repair_target_dataset[0],
                repair_target_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            rr = score[1] * 100
            score_rr.append(rr)

            # Compute BR
            repair_positive_dataset = data_util.load_repair_data(positive_inputs_dir)
            repair_images, repair_labels = (
                repair_positive_dataset[0],
                repair_positive_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            br = (1 - score[1]) * 100
            score_br.append(br)

        # Output results
        self._save_evaluate_results(
            dataset,
            model_dir,
            target_data_dir,
            positive_inputs_dir,
            output_dir,
            num_runs,
            score_rr,
            score_br,
        )
        self._log_evaluate(score_rr, score_br, num_runs, verbose)

    def evaluate(
            self,
            dataset,
            model_dir: Path,
            target_data,
            target_data_dir: Path,
            data_dict,
            age_test_label,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            fairness_measure,
            verbose=1,
    ):
        """Evaluate.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing
        verbose : int, default=1
            Log level

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []

        for i in range(num_runs):

            positive_inputs = data_dict[i]['POS']
            XA = data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(model_dir)
            print(i, 'model path: ', model_dir)

            # Localize
            localized_data_dir = output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            self.localize(model, target_data, localized_data_dir, verbose)

            # Optimize
            weights = self.load_weights(localized_data_dir_path)
            repaired_model_dir = output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()
            self.optimize(
                model,
                model_dir,
                weights,
                target_data,
                positive_inputs,
                XA,
                age_test_label,
                repaired_model_dir,
                fairness_measure,
                verbose,
            )
            # Compute RR
            model = data_util.load_model_from_tf(repaired_model_dir / "repair")
            repair_target_dataset = data_util.load_repair_data(target_data_dir)
            repair_images, repair_labels = (
                repair_target_dataset[0],
                repair_target_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            rr = score[1] * 100
            score_rr.append(rr)

            # Compute BR
            repair_positive_dataset = data_util.load_repair_data(positive_inputs_dir)
            repair_images, repair_labels = (
                repair_positive_dataset[0],
                repair_positive_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            br = (1 - score[1]) * 100
            score_br.append(br)

        # Output results
        self._save_evaluate_results(
            dataset,
            model_dir,
            target_data_dir,
            positive_inputs_dir,
            output_dir,
            num_runs,
            score_rr,
            score_br,
        )
        self._log_evaluate(score_rr, score_br, num_runs, verbose)

    def _reshape_target_model(self, model, input_neg, target_layer=None):
        """Re-shape target model for localize.

        :param model:
        :param input_neg:
        """
        if self.target_layer is None:
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

    def _compute_gradient(self, model, input_neg, desc=True):
        # For return
        candidates = []

        # Identify class of loss function
        loss_func = tf.keras.losses.get(model.loss)
        layer_index = len(model.layers) - 1
        layer = model.get_layer(index=layer_index)

        # Evaluate grad on neural weights
        with tf.GradientTape() as tape:
            # import pdb; pdb.set_trace()
            logits = model(input_neg[0])  # get the forward pass gradient
            loss_value = loss_func(input_neg[1], logits)
            grad_kernel = tape.gradient(
                loss_value, layer.kernel
            )  # TODO bias?# Evaluate grad on neural weights

        for j in trange(grad_kernel.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel.shape[0]):
                dl_dw = grad_kernel[i][j]
                # Append data tuple
                # (layer, i, j) is for identifying neural weight
                candidates.append([layer_index, i, j, np.abs(dl_dw)])

        # Sort candidates in order of grad loss
        candidates.sort(key=lambda tup: tup[3], reverse=desc)

        return candidates
    def _compute_forward_impact_seperate(self, model, input_neg, candidates, num_grad):
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
            fwd_imp = self._compute_each_forward_impact(
                model, input_neg, [layer_index, i, j], activations, w
            )
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        return pool

    def _compute_forward_impact(self, model, input_neg, candidates, num_grad):
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
            fwd_imp = self._compute_each_forward_impact(
                model, input_neg, [layer_index, i, j], activations, w
            )
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        return pool

    def _compute_each_forward_impact(self, model, input_neg, weight, activations, w):
        layer_index = weight[0]
        neural_weight_i = weight[1]
        neural_weight_j = weight[2]

        if layer_index < 1:
            raise IndexError(f"Not found previous layer: {layer_index!r}")

        o_i = activations[0][neural_weight_i]  # TODO correct?

        # Evaluate the neural weight
        w_ij = w[neural_weight_i][neural_weight_j]

        return np.abs(o_i * w_ij)

    def _extract_pareto_front(self, pool, output_dir=None, filename=r"pareto_front.png"):
        # Compute pareto front
        objectives = []
        for key in tqdm(pool, desc="Collecting objectives for pareto-front"):
            weight = pool[key]
            grad_loss = weight[3]
            fwd_imp = weight[4]
            objectives.append([grad_loss, fwd_imp])
        scores = np.array(objectives)
        pareto = self._identify_pareto(scores)
        pareto_front = scores[pareto]

        if output_dir is not None:
            self._save_pareto_front(scores, pareto_front, output_dir, filename)

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

    def _identify_pareto(self, scores):
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

    def _save_pareto_front(self, scores, pareto_front, output_dir: Path, filename: str):
        """Save pareto front in image."""
        x_all = scores[:, 0]
        y_all = scores[:, 1]
        x_pareto = pareto_front[:, 0]
        y_pareto = pareto_front[:, 1]
        # Clear the entire figure
        plt.clf()
        plt.scatter(x_all, y_all)
        plt.plot(x_pareto, y_pareto, color="r")
        plt.xlabel("Objective A")
        plt.ylabel("Objective B")

        plt.savefig(output_dir / filename)

    def _modify_layer_before_reshaped(self, orig_model, weights_t):
        """Modify the target layer to repair the original target model.

        :param orig_model:
        :param weights_t:
        """
        if self.target_layer < 0:
            target_layer = len(orig_model.layers) + self.target_layer
        else:
            target_layer = self.target_layer
        for weight in weights_t:
            weight[0] = target_layer
        return weights_t

    def _append_weights(self, model, weights):
        for weight in weights:
            layer = model.get_layer(index=int(weight[0]))
            all_weights = layer.get_weights()[0]
            weight.append(all_weights[int(weight[1])][int(weight[2])])

    def save_weights(self, weights, output_dir: Path):
        with open(output_dir / "weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(weights)
            self.output_files.add(output_dir / "weights.csv")

    def _save_evaluate_results(
            self,
            dataset,
            model_dir: Path,
            target_data_dir: Path,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            score_rr,
            score_br,
    ):
        # Compute average value
        ave_rr = sum(score_rr) / len(score_rr)
        ave_br = sum(score_br) / len(score_br)

        with open(output_dir / "result.txt", mode="w") as f:
            f.write("# Settings\n")
            f.write("dataset: %s\n" % (dataset))
            # f.write("method: %s\n" % (self.__class__.__name__))
            f.write("model_dir: %s\n" % (model_dir))
            f.write("num_grad: %s\n" % (self.num_grad))
            f.write("target_data_dir: %s\n" % (target_data_dir))
            f.write("positive_inputs_dir: %s\n" % (positive_inputs_dir))
            f.write("output_dir: %s\n" % (output_dir))
            f.write("num_particles: %s\n" % (self.num_particles))
            f.write("num_iterations: %s\n" % (self.num_iterations))
            f.write("num_runs: %s\n" % (num_runs))
            f.write("\n# Results\n")
            for i in range(num_runs):
                f.write("%d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            f.write(f"\nAverage: RR {ave_rr:.2f}%, BR {ave_br:.2f}%")

            self.output_files.add(output_dir / "result.txt")

    def _log_localize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Localize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("localized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_optimize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Optimize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("optimized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_evaluate(self, score_rr, score_br, num_runs, verbose):
        if verbose > 1:
            print("=================")
            print("Evaluate results")
            print("RR/BR")
            for i in range(num_runs):
                print("    %d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            print("=================")

    def _get_initial_particle_positions(self, weights, model):
        locations = [[0 for j in range(len(weights))] for i in range(self.num_particles)]
        for n_w in trange(len(weights), desc="Initializing particles"):
            weight = weights[n_w]
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
            samples = np.random.default_rng().normal(loc=mu, scale=std, size=self.num_particles)

            for n_p in range(self.num_particles):
                sample = samples[n_p]
                locations[n_p][n_w] = (sample, layer_index, nw_i, nw_j)

        return locations

    def _get_velocity_bounds(self, model):
        """Get velocity bounds.

        "W is the set of all neural weights
        between our target layer and the preceding one."

        wb = np.max(all_weights) - np.min(all_weights)
        vb = (wb / 5, wb * 5)

        :param model:
        :return: dictionary whose key is layer index
                 and value is velocity bounds
        """
        # Range from 1 to #layers
        velocity_bounds = {}
        for layer_index in trange(1, len(model.layers), desc="Computing velocity bounds"):
            layer = model.get_layer(index=layer_index)

            # Target only trainable layers
            if not layer.trainable:
                continue
            # Out of scope if layer does not have kernel
            if not hasattr(layer, "kernel"):
                continue

            # Get all weights
            all_weights = []
            target_weights = layer.get_weights()[0]
            for j in range(target_weights.shape[1]):
                for i in range(target_weights.shape[0]):
                    all_weights.append(target_weights[i][j])

            # Velocity bounds defined at equations 5 and 6
            wb = np.max(all_weights) - np.min(all_weights)
            vb = (wb / 5, wb * 5)

            velocity_bounds[layer_index] = vb
        return velocity_bounds

    def _sample_positive_inputs(self, input_pos):
        """Sample 200 positive inputs.

        :param input_pos:
        :return:
        """
        rg = np.random.default_rng()
        sample = rg.choice(len(input_pos[0]), self.num_input_pos_sampled)
        input_pos_sampled = (input_pos[0][sample], input_pos[1][sample], input_pos[2][sample])
        # NOTE: Temporally reverted to work with small dataset.
        # returns tuple of the sampled images from input_pos[0]
        # and their respective labels from input_pos[1]
        return input_pos_sampled, sample

    def _initialize_personal_best_scores(self, locations, model, input_pos_sampled, input_neg, XA, sensitive_input,
                                         fairness_measure):
        """Initialize personal best scores.

        :param locations: particle locations initialized
        :param model:
        :param input_pos_sampled: to compute scores
        :param input_neg: to compute scores
        :return: personal best scores
        """
        personal_best_scores = []
        personal_best_scores_with_dict = []
        for location in tqdm(locations, desc="Initializing particle's scores"):
            ## fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
            fitness, n_patched, n_intact = self._criterion(
                model, location, input_pos_sampled, input_neg, XA, sensitive_input, fairness_measure)
            ### Todo: we need to modify this with fairness,
            # _fairness_sub = self._fairness_sub(fitness_dict, measure=fairness_measure)
            ## Todo: overall_score = score - fairness constraints

            personal_best_scores.append([fitness, n_patched, n_intact])
            personal_best_scores_with_dict.append(fitness)
        return personal_best_scores, personal_best_scores_with_dict

    def fairness_measure_global(self, TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b, measure='SPD'):
        # (neg_fit + pos_fit, AOD, EOD, SPD, DI)
        f_score = 0
        if measure == 'DI':
            f_score = calculate_Disparate_Impact(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'SPD':
            f_score = calculate_SPD(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'EOD':
            f_score = calculate_equal_opportunity_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'AOD':
            f_score = calculate_average_odds_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'FPR':
            f_score = calculate_FPR_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'TPR':
            f_score = calculate_TPR_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        # else:
        return abs(f_score)

    def _fairness_metrics_global(self, TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female):
        EOD = calculate_equal_opportunity_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female,
                                                     FN_female, FP_female)
        SPD = calculate_SPD(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
        DI = calculate_Disparate_Impact(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
        AOD = calculate_average_odds_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female,
                                                FP_female)
        return abs(AOD), abs(EOD), abs(SPD), abs(DI)

    def _aggregate_val_global(self, TP, TN, FN, FP, funct=None):
        if funct is None:
            return np.mean(TP), np.mean(TN), np.mean(FN), np.mean(FP)
        else:
            return funct(TP), funct(TN), funct(FN), funct(FP)

    def _reshape_labels(self, labels):
        if len(labels.shape) > 1:
            if labels.shape[1] == 1:
                labels = np.argmax(labels, axis=1)
        return labels

    def _fairness_sub_global(self, test_labels_, prediction_, sensitive_attrib, sub_group=None, measure='DI', unfav=0):
        # print('fitness_dict: ', fitness_dict)

        # group_a, group_b = self.group_a, self.group_b
        test_labels = test_labels_.copy()
        prediction = prediction_.copy()

        prediction = self._reshape_labels(prediction)
        test_labels = self._reshape_labels(test_labels)

        if sub_group != None:
            group_a, group_b = self.group_a, self.group_b  # _get_subgroup_names_as_binary(sensitive_attrib)

            # if prediction.shape[1] > 1:
            #    prediction = np.argmax(prediction,axis=1)
            # if test_labels.shape[1] > 1:
            #    test_labels = np.argmax(test_labels,axis=1)
            TP_a, TN_a, FN_a, FP_a = [], [], [], []
            TP_b, TN_b, FN_b, FP_b = [], [], [], []
            for sub_a in group_a:
                indices = np.where(sensitive_attrib == sub_a)[0]
                pred = prediction.copy()[indices]
                y_true = test_labels.copy()[indices]
                TP_1 = ((y_true == pred) & (y_true == 1)).sum()
                TP_0 = ((y_true == pred) & (y_true == 0)).sum()
                FP_0 = ((y_true != pred) & (pred == 0)).sum()
                FP_1 = ((y_true != pred) & (pred == 1)).sum()

                TP_a.append(TP_0)
                TN_a.append(TP_1)
                FN_a.append(FP_1)
                FP_a.append(FP_0)
            for sub_b in group_b:
                indices = np.where(sensitive_attrib == sub_b)[0]
                pred = prediction.copy()[indices]
                y_true = test_labels.copy()[indices]

                TP_1 = ((y_true == pred) & (y_true == 1)).sum()
                TP_0 = ((y_true == pred) & (y_true == 0)).sum()
                FP_0 = ((y_true != pred) & (pred == 0)).sum()
                FP_1 = ((y_true != pred) & (pred == 1)).sum()

                TP_b.append(TP_1)
                TN_b.append(TP_0)
                FN_b.append(FP_0)
                FP_b.append(FP_1)
            TP_a, TN_a, FN_a, FP_a = self._aggregate_val_global(TP_a, TN_a, FN_a, FP_a, funct=np.mean)
            TP_b, TN_b, FN_b, FP_b = self._aggregate_val_global(TP_b, TN_b, FN_b, FP_b, funct=np.mean)

            if unfav == 0:
                fairness_val = self.fairness_measure_global(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b,
                                                            measure=measure)
            else:
                fairness_val = self.fairness_measure_global(TP_b, TN_b, FN_b, FP_b, TP_a, TN_a, FN_a, FP_a,
                                                            measure=measure)
        else:
            TP_1 = ((test_labels == prediction) & (test_labels == 1)).sum()
            TP_0 = ((test_labels == prediction) & (test_labels == 0)).sum()

            FP_0 = ((test_labels != prediction) & (prediction == 0)).sum()
            FP_1 = ((test_labels != prediction) & (prediction == 1)).sum()

            TN_0 = TP_1
            TN_1 = TP_0

            FN_0 = FP_1
            FN_1 = FP_0
            # TP_0, TP_1, FP_0, FP_1, FN_0, FN_1

            # fairness_val = _fairness_metrics(TP_0,TN_0,FN_0,FP_0,TP_1,TN_1,FN_1, FP_1)
            if unfav == 0:
                fairness_val = self.fairness_measure_global(TP_0, TN_0, FN_0, FP_0, TP_1, TN_1, FN_1, FP_1,
                                                            measure=measure)
            else:
                fairness_val = self.fairness_measure_global(TP_1, TN_1, FN_1, FP_1, TP_0, TN_0, FN_0, FP_0,
                                                            measure=measure)
        return fairness_val

    def _criterion(self, model, location, input_pos, input_neg, XA, sensitive_input, measure):
        # Keep original weights and set specified weights on the model
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        # for input_neg we use the generator, since it can be a big dataset.
        # For input_pos, fow now we leave the numpy array from sampling,
        # but it can be changed to also be a generator.

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        # loss_input_neg, acc_input_neg, n_patched = data_util.model_evaluate(
        #     model, input_neg, verbose=0, batch_size=self.batch_size)

        test_images, test_labels = XA[0], XA[1]  # , XA[2], XA[3]
        y_pred = model.predict(test_images, verbose=0)

        y_true = test_labels.copy()
        # y_pred = self._reshape_labels(y_pred)
        # y_true = self._reshape_labels(y_true)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)

        unfav = data_util._get_unfav(y_true, y_pred)
        #
        # N = y_true.shape[0]
        fairness_val = self._fairness_sub_global(y_true, y_pred, y_pred, measure=measure, unfav=unfav)
        # fairness_val = fairness_val*N

        # TP_1 = ((y_true == y_pred) & (y_true == 1)).sum()
        # TP_0 = ((y_true == y_pred) & (y_true == 0)).sum()
        #
        # FP_0 = ((y_true != y_pred) & (y_pred == 0)).sum()
        # FP_1 = ((y_true != y_pred) & (y_pred == 1)).sum()
        #
        # TN_0 = TP_1
        # TN_1 = TP_0
        #
        # FN_0 = FP_1
        # FN_1 = FP_0
        #
        # TP = (TP_1 + TP_0) / 2
        # TN = (TN_1 + TN_0) / 2
        #
        # FP = (FP_0 + FP_1) / 2
        # FN = (FN_0 + FN_1) / 2
        #
        # WCA = ((TP / (TP + FN + 0.001)) + (TN / (TN + FP + 0.001))) / 2
        #
        ACC = (y_true == y_pred).sum()/len(y_true)

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        # loss_input_neg, acc_input_neg, n_patched = data_util.model_evaluate(
        #     model, input_neg, verbose=0, batch_size=self.batch_size)
        #
        # # whose output is still correct"
        # loss_input_pos, acc_input_pos, n_intact = data_util.model_evaluate(
        #     model, input_pos, verbose=0, batch_size=self.batch_size
        # )
        #
        # # nt = input_neg[0].shape[0]
        # # pt = input_pos[0].shape[0]
        # ## compute the overall fitness too
        # fitness = ((n_patched + 1) / (loss_input_neg + 1)) + ((n_intact + 1) / (loss_input_pos + 1))


        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)
        # -(AOD+EOD+SPD+DI)
        # WCA-fairness_val
        return fairness_val, ACC, 'n_intact'

    def _copy_location_to_weights(self, location, model):
        for w in location:
            # Parse location data
            val = w[0]
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))

            # Set neural weight at given position with given value
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            weights[0][nw_i][nw_j] = val
            layer.set_weights(weights)

        return model

    def _copy_weights_to_location(self, model, location):
        for w in location:
            # Parse location data
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))
            # Set weight value in location with neural weight
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            w[0] = weights[0][nw_i][nw_j]

        return location

    def _update_particle(
            self,
            weights,
            v,
            velocity_bounds,
            personal_best_position,
            g,
            model,
            input_pos_sampled,
            input_neg, XA,
            sensitive_input,
            fairness_measure
    ):
        x = []
        layer_index = []
        nw_i = []
        nw_j = []
        for weight in weights:
            x.append(weight[0])
            layer_index.append(weight[1])
            nw_i.append(weight[2])
            nw_j.append(weight[3])

        # Computing new position

        ## todo: Moses Openja, commented - this is one this I need to optimize
        new_x = self._update_position(x, v)
        new_weights = []
        for n_new_x in range(len(new_x)):
            new_weights.append([new_x[n_new_x], layer_index[n_new_x], nw_i[n_new_x], nw_j[n_new_x]])

        # Computing new velocity
        p = self._get_weight_values(personal_best_position)
        new_v = self._update_velocity(new_x, v, p, g, velocity_bounds, layer_index)

        # Computing new score
        # fitness_dict, fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
        score, n_patched, n_intact = self._criterion(
            model, new_weights, input_pos_sampled, input_neg, XA, sensitive_input, fairness_measure,
        )
        # score, n_patched, n_intact = self._criterion(
        #     model, new_weights, input_pos_sampled, input_neg, sensitive_input,
        # )

        return new_weights, new_v, score, n_patched, n_intact

    def _get_weight_values(self, weights):
        values = []
        for w in weights:
            values.append(w[0])
        return values

    def _update_position(self, x, v):
        return x + v

    def _update_velocity(self, x, v, p, g, vb, layer_index):
        # "We follow the general recommendation
        # in the literature and set both to 4.1"
        phi = self.velocity_phi
        # "Equation 3"
        chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
        # "Equation 2"
        ro1 = random.uniform(0, phi)
        ro2 = random.uniform(0, phi)
        # TODO Using same value 'chi'
        #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
        new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))
        "we additionally set velocity bounds"
        for n in range(len(new_v)):
            _vb = vb[layer_index[n]]
            _new_v = np.abs(new_v[n])
            _sign = 1 if 0 < new_v[n] else -1
            if _new_v < _vb[0]:
                new_v[n] = _vb[0] * _sign
            if _vb[1] < _new_v:
                new_v[n] = _vb[1] * _sign
        return new_v

    def _fail_to_find_better_patch(self, t, history):
        # "stop earlier if it fails to find a better patch
        # than the current best during ten consecutive iterations"
        if self.min_iteration_range < t:
            scores_in_history = np.array(history)[:, 0]

            best_x_before = scores_in_history[-self.min_iteration_range - 1]
            # Get the current best during ten consecutive iterations
            best_last_x = max(scores_in_history[-self.min_iteration_range:])

            # found a better patch, continue PSO
            if best_last_x > best_x_before:
                return False
            # fail to find a better patch, stagnated, stop PSO
            else:
                return True
        else:
            return False

    def _fail_to_find_better_patch_fairness(self, t, history):
        # "stop earlier if it fails to find a better patch
        # than the current best during ten consecutive iterations"
        if self.min_iteration_range < t:
            scores_in_history = np.array(history)[:, 0]

            best_x_before = scores_in_history[-self.min_iteration_range - 1]
            # Get the current best during ten consecutive iterations
            best_last_x = min(scores_in_history[-self.min_iteration_range:])

            # found a better patch, continue PSO
            if best_last_x > best_x_before:
                return False
            # fail to find a better patch, stagnated, stop PSO
            else:
                return True
        else:
            return False

    def _output_repaired_model(self, output_dir, model_repaired):
        # Output
        output_dir = Path(output_dir)
        # Clean directory
        repair_dir = output_dir / "repair"
        if repair_dir.exists():
            shutil.rmtree(repair_dir)
        repair_dir.mkdir()
        # Save model
        model_repaired.save(repair_dir)

    def load_weights(self, weights_dir: Path):
        candidates = []
        with open(weights_dir / "weights.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                candidates.append(row[:3])
        return candidates


##todo: Normal arachne
class Arachne:
    def __init__(self):
        """Initialize."""
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 30
        self.num_input_pos_sampled = 200
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = None
        self.output_files = set()
        self.batch_size = 32
        #self.random_seed = 0
        #np.random.seed(self.random_seed)
    def localize_bias(self, model_dir, input_neg, input_pos, input_both, output_dir: Path, verbose=1, unfav=0, sample_global=False, based_on=2, topN=10):
        print(input_neg[0].shape, self.num_grad, self.target_layer)
        model = data_util.load_model_from_tf(Path(model_dir))
        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20
        ## todo: modified by moses, to avoid processing output layer
        if self.target_layer == len(model.layers) - 1:
            self.target_layer = None
        ## todo: end of updates by moses
        fault_localization = FaultLocalization(model_dir, input_neg, input_pos, input_both, unfav=unfav,
                                               sample_global=sample_global)

        reshaped_model = self._reshape_target_model(model, input_neg)
        if based_on == 0:
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_neg, based_on=based_on)
            _,pool = fault_localization._compute_forward_impact_single(reshaped_model, input_neg, candidates, list_candidates, data_candidate_dict, self.num_grad, based_on=based_on)
            # candidates = self._compute_gradient(reshaped_model, input_neg)
            # pool = self._compute_forward_impact(reshaped_model, input_neg, candidates, self.num_grad)
        elif based_on == 1:
            #candidates = fault_localization._compute_gradient_pos_neg(reshaped_model)
            list_imp_neg, candidates_neg, data_candidate_dict_neg = fault_localization._compute_gradient(reshaped_model, input_neg)
            list_imp_pos, candidates_pos, data_candidate_dict_pos = fault_localization._compute_gradient(reshaped_model, input_pos,
                                                                                based_on=based_on)
            list_pool, pool = fault_localization._compute_forward_impact_neg_pos(reshaped_model, (candidates_neg, candidates_pos), (list_imp_neg, list_imp_pos), (data_candidate_dict_neg, data_candidate_dict_pos), self.num_grad, based_on=based_on)
        elif based_on == 2: ## Both positive and negative, without knowledge of sensitive
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model, input_both, based_on=based_on)
            _, pool = fault_localization._compute_forward_impact_single(reshaped_model, input_both, candidates, list_candidates, data_candidate_dict, self.num_grad, based_on=based_on)
        elif based_on == 3: ## Arachne v2,
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient_arachne_v2(reshaped_model,
                                                                                                    based_on=based_on)
            _, pool = fault_localization._compute_forward_impact_arachne_v2(reshaped_model, candidates,
                                                                        list_candidates, data_candidate_dict,
                                                                        self.num_grad, based_on=based_on)
        elif based_on == -2:
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient(reshaped_model,
                                                                                                    input_both,
                                                                                                    based_on=2)
            weights_t = candidates[:topN]
        elif based_on == -3:
            list_candidates, candidates, data_candidate_dict = fault_localization._compute_gradient_arachne_v2(
                reshaped_model,
                based_on=3)
            weights_t = candidates[:topN]
        if based_on == 1 or based_on == 2 or based_on == 3:
            weights_t = self._extract_pareto_front(pool, output_dir)
            weights_t = self._modify_layer_before_reshaped(model, weights_t)
            #print('1: ', len(pool), len(candidates), len(weights_t))
            # Output neural weight candidates to repair
            self._append_weights(model, weights_t)
        self.save_weights(weights_t, output_dir)
        #print('2: ', len(pool), len(candidates), len(weights_t))
        self._log_localize(weights_t, verbose)
        return weights_t

    def localize(self, model, input_neg, output_dir: Path, verbose=1):
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
        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20
        ## todo: modified by moses, to avoid processing output layer
        if self.target_layer == len(model.layers) - 1:
            self.target_layer = None
        ## todo: end of updates by moses

        reshaped_model = self._reshape_target_model(model, input_neg)

        candidates = self._compute_gradient(reshaped_model, input_neg)

        pool = self._compute_forward_impact(reshaped_model, input_neg, candidates, self.num_grad)

        weights_t = self._extract_pareto_front(pool, output_dir)

        weights_t = self._modify_layer_before_reshaped(model, weights_t)

        # Output neural weight candidates to repair
        self._append_weights(model, weights_t)
        self.save_weights(weights_t, output_dir)

        self._log_localize(weights_t, verbose)
        return weights_t

    def _get_subgroup_names_as_binary(self, sensitive_input):
        # keys = list(fitness_dict.keys())
        sensitive_unique = np.unique(sensitive_input)
        sensitive_unique.sort()
        self.group_a, self.group_b = [], []
        for i in range(int(len(sensitive_unique) / 2)):
            self.group_a.append(sensitive_unique[i])
        for i in range(int(len(sensitive_unique) / 2), len(sensitive_unique)):
            self.group_b.append(sensitive_unique[i])
        return self.group_a, self.group_b

    def optimize(
            self,
            model,
            model_dir: Path,
            weights,
            input_neg,
            input_pos,
            XA,
            sensitive_input,
            output_dir: Path,
            fairness_measure='DI',
            verbose=1,
    ):
        """Optimize.

        cf. https://qiita.com/sz_dr/items/bccb478965195c5e4097

        Parameters
        ----------
        model :
            DNN model to repair
        model_dir : Path
            (Not used)
        weights
            Set of neural weights to target for repair
        input_neg
            Dataset for unexpected behavior
        input_pos
            Dataset for correct behavior
        output_dir : Path
            Path to directory to save result
        verbose : int, default=1
            Log level

        """
        group_a, group_b = self._get_subgroup_names_as_binary(sensitive_input)
        # Initialize particle positions
        locations = self._get_initial_particle_positions(weights, model)

        # "The initial velocity of each particle is set to zero"
        velocities = np.zeros((self.num_particles, len(weights)))

        # Compute velocity bounds
        velocity_bounds = self._get_velocity_bounds(model)

        input_pos_sampled = input_pos
        input_neg_sampled = input_neg
        # "We sample 200 positive inputs"
        #input_pos_sampled, pos_sample = self._sample_positive_inputs(input_pos)

        #input_neg_sampled, neg_sample = self._sample_positive_inputs(input_neg)
        # Convert the dataset into numpy-format not to use generators of tensorflow.
        # When the number of dataset is not so large,
        # the cost of making generators can not be ignored, and it causes memory-leak.
        # neg_sample = input_neg.sample_from_file(input_pos_sampled[0].shape[0])
        # input_neg_sampled = (input_neg[0][neg_sample], input_neg[1][neg_sample], input_neg[2][neg_sample])
        sensitive_input_org = sensitive_input.copy()
        # Initialize for PSO search
        personal_best_positions = list(locations)

        # for n_p in range(self.num_particles):
        #     sample = samples[n_p]
        #     locations[n_p][n_w] = (sample, layer_index, nw_i, nw_j)

        personal_best_scores, _ = self._initialize_personal_best_scores(
            locations, model, input_pos_sampled, input_neg_sampled, XA, sensitive_input_org,
            fairness_measure=fairness_measure
        )
        ## todo: we should update here too!
        # fitness_dict, fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
        ## convert to binary
        # fitness_dict[key] = ((neg_fit + pos_fit), (AOD, EOD, SPD, DI))
        # list_abs_fit = self._abs_fitness_list(personal_best_scores_with_dict, personal_best_scores)
        best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
        # best_particle = np.argmax(np.array(list_abs_fit))
        global_best_position = personal_best_positions[best_particle]

        # Search
        history = []
        pef_history_global = []
        # "PSO uses ... the maximum number of iterations is 100"
        for t in range(self.num_iterations):
            g = self._get_weight_values(global_best_position)
            # "PSO uses a population size of 100"

            # fitness_dict_list = []
            score_list = []
            n_patched_list = []

            pef_history = []
            n_intact_list = []
            for n in trange(
                    self.num_particles,
                    desc="Updating particle positions" f" (it={t + 1}/{self.num_iterations})",
            ):
                new_weights, new_v, score, f_patched, n_intact = self._update_particle(
                    locations[n],
                    velocities[n],
                    velocity_bounds,
                    personal_best_positions[n],
                    g,
                    model,
                    input_pos_sampled,
                    input_neg_sampled,
                    XA,
                    sensitive_input_org,
                    fairness_measure
                )

                # fitness_dict_list.append(fitness_dict)
                score_list.append(score)
                n_patched_list.append(f_patched)
                n_intact_list.append(n_intact)
                # Update position
                locations[n] = new_weights
                # Update velocity
                velocities[n] = new_v

                # _fairness_sub = self._fairness_sub(fitness_dict, measure=fairness_measure)
                ## Todo: overall_score = score -  fairness constraints
                # list_abs_fit.append(fitness-abs(_fairness_sub))
                # overall_score = _fairness_sub #score-abs(_fairness_sub)

                # Update score
                if personal_best_scores[n][0] < score:
                    # print(personal_best_scores[n][0], score)
                    personal_best_scores[n] = [score, f_patched, n_intact]
                    personal_best_positions[n] = locations[n]
                    pef_history.append([score, f_patched, n_intact])
                    pef_history_global.append([score, f_patched, n_intact])
            # list_abs_fit = self._abs_fitness_list(fitness_dict_list, score_list)
            # Update global best

            # if t%10 == 0:
            #     self.save_metrics(pef_history, t,'SPD',output_dir)
            #     self.plot_performance(pef_history, t, output_dir)
            # best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
            best_particle = np.argmax(np.array(personal_best_scores)[:, 0])
            # best_particle = np.argmax(np.array(list_abs_fit))
            global_best_position = personal_best_positions[best_particle]
            # Add current best
            history.append(personal_best_scores[best_particle])

            # Stop earlier
            if self._fail_to_find_better_patch(t, history):
                break
            # if self._fail_to_find_better_patch_fairness(t, history):
            #     break
        # self.save_metrics(pef_history_global, 0, 'SPD', output_dir)
        # self.plot_performance(pef_history_global, 0, output_dir)

        self._append_weights(model, weights)
        model = self._copy_location_to_weights(global_best_position, model)
        self._append_weights(model, weights)
        self.save_weights(weights, output_dir)

        self._output_repaired_model(output_dir, model)
        self._log_optimize(global_best_position, verbose)

        return model

    def save_metrics(self, history, epoch, fairness, output_dir):
        with open(output_dir / "metrics-{}-{}.csv".format(fairness, epoch), "w") as f:
            writer = csv.writer(f)
            writer.writerows(history)
    def plot_performance(self, history, epoch, output_dir, fairness='SPD'):
        x = [i for i in range(len(history))]
        # y1 = [1000, 13000, 26000, 42000, 60000, 81000]
        # y2 = [1000, 13000, 27000, 43000, 63000, 85000]
        history = np.array(history)
        # Clear the entire figure
        plt.clf()
        #plt.plot(x, history[:, 0], label='arachne')
        plt.plot(x, history[:, 1], label=fairness)
        #plt.plot(x, history[:, 2], '-.', label='ACC')

        # plt.xlabel("X-axis data")
        # plt.ylabel("Y-axis data")
        plt.legend()
        # plt.title('multiple plots')
        plt.savefig(output_dir / 'metric-{}-{}.png'.format(fairness, epoch), bbox_inches='tight')
        #plt.show()
        # Close the figure
        plt.close()

    def evaluate(
            self,
            dataset,
            model_dir: Path,
            # target_data,
            target_data_dir: Path,
            data_dict,
            age_test_label,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            fairness_measure,
            unfav,
            sample_global,
            based_on,
            verbose=1,
    ):
        """Evaluate.

        Parameters
        ----------
        dataset :
            Dataset instance
        model_dir : Path
            Path to directory containing model files
        target_data:
            Negative dataset
        target_data_dir : Path
            Path to directory containing negative dataset
        positive_inputs :
            Positive dataset
        positive_inputs_dir : Path
            Path to directory containing positive dataset
        output_dir : Path
            Path to directory to save results
        num_runs : int
            Number of iterations for repairing
        verbose : int, default=1
            Log level

        """
        # Make output directory if not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        score_rr = []
        score_br = []

        for i in range(num_runs):
            input_neg = data_dict[i]['POS']
            input_pos = data_dict[i]['POS']
            input_both = data_dict[i]['BOTH']
            # Load
            model = data_util.load_model_from_tf(model_dir)
            print(i, 'model path: ', model_dir)

            # Localize
            localized_data_dir = output_dir / f"localized_data_{i}"
            localized_data_dir_path = localized_data_dir
            if localized_data_dir_path.exists():
                shutil.rmtree(localized_data_dir_path)
            localized_data_dir_path.mkdir()
            self.output_files = set()
            self.localize_bias(model_dir, input_neg, input_pos, input_both, localized_data_dir, verbose, unfav=unfav,
                               sample_global=False, based_on=based_on)
            #self.localize_bias(model_dir, target_data, localized_data_dir, verbose)

            # Optimize
            weights = self.load_weights(localized_data_dir_path)
            repaired_model_dir = output_dir / f"repaired_model_{i}"
            if repaired_model_dir.exists():
                shutil.rmtree(repaired_model_dir)
            repaired_model_dir.mkdir()
            self.output_files = set()
            self.optimize(
                model,
                model_dir,
                weights,
                input_neg,
                input_pos,
                input_both,
                age_test_label,
                repaired_model_dir,
                fairness_measure,
                verbose,
            )
            # Compute RR
            model = data_util.load_model_from_tf(repaired_model_dir / "repair")
            repair_target_dataset = data_util.load_repair_data(target_data_dir)
            repair_images, repair_labels = (
                repair_target_dataset[0],
                repair_target_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            rr = score[1] * 100
            score_rr.append(rr)

            # Compute BR
            repair_positive_dataset = data_util.load_repair_data(positive_inputs_dir)
            repair_images, repair_labels = (
                repair_positive_dataset[0],
                repair_positive_dataset[1],
            )

            score = model.evaluate(
                repair_images, repair_labels, verbose=0, batch_size=self.batch_size
            )
            br = (1 - score[1]) * 100
            score_br.append(br)

        # Output results
        self._save_evaluate_results(
            dataset,
            model_dir,
            target_data_dir,
            positive_inputs_dir,
            output_dir,
            num_runs,
            score_rr,
            score_br,
        )
        self._log_evaluate(score_rr, score_br, num_runs, verbose)

    def _reshape_target_model(self, model, input_neg, target_layer=None):
        """Re-shape target model for localize.

        :param model:
        :param input_neg:
        """
        if self.target_layer is None:
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

    def _compute_gradient(self, model, input_neg, desc=True):
        # For return
        candidates = []

        # Identify class of loss function
        loss_func = tf.keras.losses.get(model.loss)
        layer_index = len(model.layers) - 1
        layer = model.get_layer(index=layer_index)

        # Evaluate grad on neural weights
        with tf.GradientTape() as tape:
            # import pdb; pdb.set_trace()
            logits = model(input_neg[0])  # get the forward pass gradient
            loss_value = loss_func(input_neg[1], logits)
            grad_kernel = tape.gradient(
                loss_value, layer.kernel
            )  # TODO bias?# Evaluate grad on neural weights

        for j in trange(grad_kernel.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel.shape[0]):
                dl_dw = grad_kernel[i][j]
                # Append data tuple
                # (layer, i, j) is for identifying neural weight
                candidates.append([layer_index, i, j, np.abs(dl_dw)])

        # Sort candidates in order of grad loss
        candidates.sort(key=lambda tup: tup[3], reverse=desc)

        return candidates
    def _compute_forward_impact_seperate(self, model, input_neg, candidates, num_grad):
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
            fwd_imp = self._compute_each_forward_impact(
                model, input_neg, [layer_index, i, j], activations, w
            )
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        return pool

    def _compute_forward_impact(self, model, input_neg, candidates, num_grad):
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
            fwd_imp = self._compute_each_forward_impact(
                model, input_neg, [layer_index, i, j], activations, w
            )
            pool[num] = [layer_index, i, j, grad_loss, fwd_imp]
        return pool

    def _compute_each_forward_impact(self, model, input_neg, weight, activations, w):
        layer_index = weight[0]
        neural_weight_i = weight[1]
        neural_weight_j = weight[2]

        if layer_index < 1:
            raise IndexError(f"Not found previous layer: {layer_index!r}")

        o_i = activations[0][neural_weight_i]  # TODO correct?

        # Evaluate the neural weight
        w_ij = w[neural_weight_i][neural_weight_j]

        return np.abs(o_i * w_ij)

    def _extract_pareto_front(self, pool, output_dir=None, filename=r"pareto_front.png"):
        # Compute pareto front
        objectives = []
        for key in tqdm(pool, desc="Collecting objectives for pareto-front"):
            weight = pool[key]
            grad_loss = weight[3]
            fwd_imp = weight[4]
            objectives.append([grad_loss, fwd_imp])
        scores = np.array(objectives)
        pareto = self._identify_pareto(scores)
        pareto_front = scores[pareto]

        if output_dir is not None:
            self._save_pareto_front(scores, pareto_front, output_dir, filename)

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

    def _identify_pareto(self, scores):
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

    def _save_pareto_front(self, scores, pareto_front, output_dir: Path, filename: str):
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

    def _modify_layer_before_reshaped(self, orig_model, weights_t):
        """Modify the target layer to repair the original target model.

        :param orig_model:
        :param weights_t:
        """
        if self.target_layer < 0:
            target_layer = len(orig_model.layers) + self.target_layer
        else:
            target_layer = self.target_layer
        for weight in weights_t:
            weight[0] = target_layer
        return weights_t

    def _append_weights(self, model, weights):
        for weight in weights:
            layer = model.get_layer(index=int(weight[0]))
            all_weights = layer.get_weights()[0]
            weight.append(all_weights[int(weight[1])][int(weight[2])])

    def save_weights(self, weights, output_dir: Path):
        with open(output_dir / "weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(weights)
            self.output_files.add(output_dir / "weights.csv")

    def _save_evaluate_results(
            self,
            dataset,
            model_dir: Path,
            target_data_dir: Path,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
            score_rr,
            score_br,
    ):
        # Compute average value
        ave_rr = sum(score_rr) / len(score_rr)
        ave_br = sum(score_br) / len(score_br)

        with open(output_dir / "result.txt", mode="w") as f:
            f.write("# Settings\n")
            f.write("dataset: %s\n" % (dataset))
            # f.write("method: %s\n" % (self.__class__.__name__))
            f.write("model_dir: %s\n" % (model_dir))
            f.write("num_grad: %s\n" % (self.num_grad))
            f.write("target_data_dir: %s\n" % (target_data_dir))
            f.write("positive_inputs_dir: %s\n" % (positive_inputs_dir))
            f.write("output_dir: %s\n" % (output_dir))
            f.write("num_particles: %s\n" % (self.num_particles))
            f.write("num_iterations: %s\n" % (self.num_iterations))
            f.write("num_runs: %s\n" % (num_runs))
            f.write("\n# Results\n")
            for i in range(num_runs):
                f.write("%d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            f.write(f"\nAverage: RR {ave_rr:.2f}%, BR {ave_br:.2f}%")

            self.output_files.add(output_dir / "result.txt")

    def _log_localize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Localize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("localized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_optimize(self, results, verbose):
        if verbose > 0:
            print("=================")
            print("Optimize results")
            print("created files:")
            for file_path in self.output_files:
                print("    ", file_path)
            if verbose > 1:
                print("optimized weights:")
                for result in results:
                    print("    ", result)
            print("=================")

    def _log_evaluate(self, score_rr, score_br, num_runs, verbose):
        if verbose > 1:
            print("=================")
            print("Evaluate results")
            print("RR/BR")
            for i in range(num_runs):
                print("    %d: RR %.2f%%, BR %.2f%%\n" % (i, score_rr[i], score_br[i]))
            print("=================")

    def _get_initial_particle_positions(self, weights, model):
        locations = [[0 for j in range(len(weights))] for i in range(self.num_particles)]
        for n_w in trange(len(weights), desc="Initializing particles"):
            weight = weights[n_w]
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
            samples = np.random.default_rng().normal(loc=mu, scale=std, size=self.num_particles)

            for n_p in range(self.num_particles):
                sample = samples[n_p]
                locations[n_p][n_w] = (sample, layer_index, nw_i, nw_j)

        return locations

    def _get_velocity_bounds(self, model):
        """Get velocity bounds.

        "W is the set of all neural weights
        between our target layer and the preceding one."

        wb = np.max(all_weights) - np.min(all_weights)
        vb = (wb / 5, wb * 5)

        :param model:
        :return: dictionary whose key is layer index
                 and value is velocity bounds
        """
        # Range from 1 to #layers
        velocity_bounds = {}
        for layer_index in trange(1, len(model.layers), desc="Computing velocity bounds"):
            layer = model.get_layer(index=layer_index)

            # Target only trainable layers
            if not layer.trainable:
                continue
            # Out of scope if layer does not have kernel
            if not hasattr(layer, "kernel"):
                continue

            # Get all weights
            all_weights = []
            target_weights = layer.get_weights()[0]
            for j in range(target_weights.shape[1]):
                for i in range(target_weights.shape[0]):
                    all_weights.append(target_weights[i][j])

            # Velocity bounds defined at equations 5 and 6
            wb = np.max(all_weights) - np.min(all_weights)
            vb = (wb / 5, wb * 5)

            velocity_bounds[layer_index] = vb
        return velocity_bounds

    def _sample_positive_inputs(self, input_pos):
        """Sample 200 positive inputs.

        :param input_pos:
        :return:
        """
        rg = np.random.default_rng()
        sample = rg.choice(len(input_pos[0]), self.num_input_pos_sampled)
        input_pos_sampled = (input_pos[0][sample], input_pos[1][sample], input_pos[2][sample])
        # NOTE: Temporally reverted to work with small dataset.
        # returns tuple of the sampled images from input_pos[0]
        # and their respective labels from input_pos[1]
        return input_pos_sampled, sample

    def _initialize_personal_best_scores(self, locations, model, input_pos_sampled, input_neg, XA, sensitive_input,
                                         fairness_measure):
        """Initialize personal best scores.

        :param locations: particle locations initialized
        :param model:
        :param input_pos_sampled: to compute scores
        :param input_neg: to compute scores
        :return: personal best scores
        """
        personal_best_scores = []
        personal_best_scores_with_dict = []
        for location in tqdm(locations, desc="Initializing particle's scores"):
            ## fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
            fitness, n_patched, n_intact = self._criterion(
                model, location, input_pos_sampled, input_neg, XA, sensitive_input, fairness_measure)
            ### Todo: we need to modify this with fairness,
            # _fairness_sub = self._fairness_sub(fitness_dict, measure=fairness_measure)
            ## Todo: overall_score = score - fairness constraints

            personal_best_scores.append([fitness, n_patched, n_intact])
            personal_best_scores_with_dict.append(fitness)
        return personal_best_scores, personal_best_scores_with_dict

    def fairness_measure_global(self, TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b, measure='SPD'):
        # (neg_fit + pos_fit, AOD, EOD, SPD, DI)
        f_score = 0
        if measure == 'DI':
            f_score = calculate_Disparate_Impact(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'SPD':
            f_score = calculate_SPD(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'EOD':
            f_score = calculate_equal_opportunity_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        elif measure == 'AOD':
            f_score = calculate_average_odds_difference(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b)
        # else:
        return abs(f_score)

    def _fairness_metrics_global(self, TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female):
        EOD = calculate_equal_opportunity_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female,
                                                     FN_female, FP_female)
        SPD = calculate_SPD(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
        DI = calculate_Disparate_Impact(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female, FP_female)
        AOD = calculate_average_odds_difference(TP_male, TN_male, FN_male, FP_male, TP_female, TN_female, FN_female,
                                                FP_female)
        return abs(AOD), abs(EOD), abs(SPD), abs(DI)

    def _aggregate_val_global(self, TP, TN, FN, FP, funct=None):
        if funct is None:
            return np.mean(TP), np.mean(TN), np.mean(FN), np.mean(FP)
        else:
            return funct(TP), funct(TN), funct(FN), funct(FP)

    def _reshape_labels(self, labels):
        if len(labels.shape) > 1:
            if labels.shape[1] == 1:
                labels = np.argmax(labels, axis=1)
        return labels

    def _fairness_sub_global(self, test_labels_, prediction_, sensitive_attrib, sub_group=None, measure='DI', unfav=0):
        # print('fitness_dict: ', fitness_dict)

        # group_a, group_b = self.group_a, self.group_b
        test_labels = test_labels_.copy()
        prediction = prediction_.copy()

        prediction = self._reshape_labels(prediction)
        test_labels = self._reshape_labels(test_labels)

        if sub_group != None:
            group_a, group_b = self.group_a, self.group_b  # _get_subgroup_names_as_binary(sensitive_attrib)

            # if prediction.shape[1] > 1:
            #    prediction = np.argmax(prediction,axis=1)
            # if test_labels.shape[1] > 1:
            #    test_labels = np.argmax(test_labels,axis=1)
            TP_a, TN_a, FN_a, FP_a = [], [], [], []
            TP_b, TN_b, FN_b, FP_b = [], [], [], []
            for sub_a in group_a:
                indices = np.where(sensitive_attrib == sub_a)[0]
                pred = prediction.copy()[indices]
                y_true = test_labels.copy()[indices]
                TP_1 = ((y_true == pred) & (y_true == 1)).sum()
                TP_0 = ((y_true == pred) & (y_true == 0)).sum()
                FP_0 = ((y_true != pred) & (pred == 0)).sum()
                FP_1 = ((y_true != pred) & (pred == 1)).sum()

                TP_a.append(TP_0)
                TN_a.append(TP_1)
                FN_a.append(FP_1)
                FP_a.append(FP_0)
            for sub_b in group_b:
                indices = np.where(sensitive_attrib == sub_b)[0]
                pred = prediction.copy()[indices]
                y_true = test_labels.copy()[indices]

                TP_1 = ((y_true == pred) & (y_true == 1)).sum()
                TP_0 = ((y_true == pred) & (y_true == 0)).sum()
                FP_0 = ((y_true != pred) & (pred == 0)).sum()
                FP_1 = ((y_true != pred) & (pred == 1)).sum()

                TP_b.append(TP_1)
                TN_b.append(TP_0)
                FN_b.append(FP_0)
                FP_b.append(FP_1)
            TP_a, TN_a, FN_a, FP_a = self._aggregate_val_global(TP_a, TN_a, FN_a, FP_a, funct=np.mean)
            TP_b, TN_b, FN_b, FP_b = self._aggregate_val_global(TP_b, TN_b, FN_b, FP_b, funct=np.mean)

            if unfav == 0:
                fairness_val = self.fairness_measure_global(TP_a, TN_a, FN_a, FP_a, TP_b, TN_b, FN_b, FP_b,
                                                            measure=measure)
            else:
                fairness_val = self.fairness_measure_global(TP_b, TN_b, FN_b, FP_b, TP_a, TN_a, FN_a, FP_a,
                                                            measure=measure)
        else:
            TP_1 = ((test_labels == prediction) & (test_labels == 1)).sum()
            TP_0 = ((test_labels == prediction) & (test_labels == 0)).sum()

            FP_0 = ((test_labels != prediction) & (prediction == 0)).sum()
            FP_1 = ((test_labels != prediction) & (prediction == 1)).sum()

            TN_0 = TP_1
            TN_1 = TP_0

            FN_0 = FP_1
            FN_1 = FP_0
            # TP_0, TP_1, FP_0, FP_1, FN_0, FN_1

            # fairness_val = _fairness_metrics(TP_0,TN_0,FN_0,FP_0,TP_1,TN_1,FN_1, FP_1)
            if unfav == 0:
                fairness_val = self.fairness_measure_global(TP_0, TN_0, FN_0, FP_0, TP_1, TN_1, FN_1, FP_1,
                                                            measure=measure)
            else:
                fairness_val = self.fairness_measure_global(TP_1, TN_1, FN_1, FP_1, TP_0, TN_0, FN_0, FP_0,
                                                            measure=measure)
        return fairness_val

    def _criterion(self, model, location, input_pos, input_neg, XA, sensitive_input, measure):
        # Keep original weights and set specified weights on the model
        orig_location = np.copy(location)
        orig_location = self._copy_weights_to_location(model, orig_location)
        model = self._copy_location_to_weights(location, model)

        test_images, test_labels = XA[0], XA[1]  # , XA[2], XA[3]
        y_pred = model.predict(test_images, verbose=0)

        y_true = test_labels.copy()
        # y_pred = self._reshape_labels(y_pred)
        # y_true = self._reshape_labels(y_true)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)

        unfav = data_util._get_unfav(y_true, y_pred)
        #
        # N = y_true.shape[0]
        fairness_val = self._fairness_sub_global(y_true, y_pred, y_pred, measure='SPD', unfav=unfav)

        # for input_neg we use the generator, since it can be a big dataset.
        # For input_pos, fow now we leave the numpy array from sampling,
        # but it can be changed to also be a generator.

        # "N_{patched} is the number of inputs in I_{neg}
        # whose output is corrected by the current patch"
        loss_input_neg, acc_input_neg, n_patched = data_util.model_evaluate(
            model, input_neg, verbose=0, batch_size=self.batch_size)

        # test_images, test_labels = XA[0], XA[1]  # , XA[2], XA[3]
        # y_pred = model.predict(test_images, verbose=0)
        #
        # y_true = test_labels.copy()
        # y_pred = self._reshape_labels(y_pred)
        # y_true = self._reshape_labels(y_true)
        # #
        # # N = y_true.shape[0]
        # fairness_val = self._fairness_sub_global(y_true, y_pred, y_pred, measure=measure)
        # fairness_val = fairness_val*N

        # whose output is still correct"
        loss_input_pos, acc_input_pos, n_intact = data_util.model_evaluate(
            model, input_pos, verbose=0, batch_size=self.batch_size
        )

        # nt = input_neg[0].shape[0]
        # pt = input_pos[0].shape[0]
        ## compute the overall fitness too
        fitness = ((n_patched + 1) / (loss_input_neg + 1)) + ((n_intact + 1) / (loss_input_pos + 1))

        #print('fitness: ', fitness, fairness_val, fitness/4)
        # Restore original weights to the model
        model = self._copy_location_to_weights(orig_location, model)
        # -(AOD+EOD+SPD+DI)

        return fitness, fairness_val, 'n_intact'

    def _copy_location_to_weights(self, location, model):
        for w in location:
            # Parse location data
            val = w[0]
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))

            # Set neural weight at given position with given value
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            weights[0][nw_i][nw_j] = val
            layer.set_weights(weights)

        return model

    def _copy_weights_to_location(self, model, location):
        for w in location:
            # Parse location data
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))
            # Set weight value in location with neural weight
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            w[0] = weights[0][nw_i][nw_j]

        return location

    def _update_particle(
            self,
            weights,
            v,
            velocity_bounds,
            personal_best_position,
            g,
            model,
            input_pos_sampled,
            input_neg, XA,
            sensitive_input,
            fairness_measure
    ):
        x = []
        layer_index = []
        nw_i = []
        nw_j = []
        for weight in weights:
            x.append(weight[0])
            layer_index.append(weight[1])
            nw_i.append(weight[2])
            nw_j.append(weight[3])

        # Computing new position

        ## todo: Moses Openja, commented - this is one this I need to optimize
        new_x = self._update_position(x, v)
        new_weights = []
        for n_new_x in range(len(new_x)):
            new_weights.append([new_x[n_new_x], layer_index[n_new_x], nw_i[n_new_x], nw_j[n_new_x]])

        # Computing new velocity
        p = self._get_weight_values(personal_best_position)
        new_v = self._update_velocity(new_x, v, p, g, velocity_bounds, layer_index)

        # Computing new score
        # fitness_dict, fitness, n_patched, n_intact, indices_neg_dict, indices_pos_dict
        score, n_patched, n_intact = self._criterion(
            model, new_weights, input_pos_sampled, input_neg, XA, sensitive_input, fairness_measure,
        )
        # score, n_patched, n_intact = self._criterion(
        #     model, new_weights, input_pos_sampled, input_neg, sensitive_input,
        # )

        return new_weights, new_v, score, n_patched, n_intact

    def _get_weight_values(self, weights):
        values = []
        for w in weights:
            values.append(w[0])
        return values

    def _update_position(self, x, v):
        return x + v

    def _update_velocity(self, x, v, p, g, vb, layer_index):
        # "We follow the general recommendation
        # in the literature and set both to 4.1"
        phi = self.velocity_phi
        # "Equation 3"
        chi = 2 / (phi - 2 + np.sqrt(phi * phi - 4 * phi))
        # "Equation 2"
        ro1 = random.uniform(0, phi)
        ro2 = random.uniform(0, phi)
        # TODO Using same value 'chi'
        #  to 'w', 'c1', and 'c2' in PSO hyper-parameters?
        new_v = chi * (v + ro1 * (p - x) + ro2 * (g - x))
        "we additionally set velocity bounds"
        for n in range(len(new_v)):
            _vb = vb[layer_index[n]]
            _new_v = np.abs(new_v[n])
            _sign = 1 if 0 < new_v[n] else -1
            if _new_v < _vb[0]:
                new_v[n] = _vb[0] * _sign
            if _vb[1] < _new_v:
                new_v[n] = _vb[1] * _sign
        return new_v

    def _fail_to_find_better_patch(self, t, history):
        # "stop earlier if it fails to find a better patch
        # than the current best during ten consecutive iterations"
        if self.min_iteration_range < t:
            scores_in_history = np.array(history)[:, 0]

            best_x_before = scores_in_history[-self.min_iteration_range - 1]
            # Get the current best during ten consecutive iterations
            best_last_x = max(scores_in_history[-self.min_iteration_range:])

            # found a better patch, continue PSO
            if best_last_x > best_x_before:
                return False
            # fail to find a better patch, stagnated, stop PSO
            else:
                return True
        else:
            return False

    def _fail_to_find_better_patch_fairness(self, t, history):
        # "stop earlier if it fails to find a better patch
        # than the current best during ten consecutive iterations"
        if self.min_iteration_range < t:
            scores_in_history = np.array(history)[:, 0]

            best_x_before = scores_in_history[-self.min_iteration_range - 1]
            # Get the current best during ten consecutive iterations
            best_last_x = min(scores_in_history[-self.min_iteration_range:])

            # found a better patch, continue PSO
            if best_last_x > best_x_before:
                return False
            # fail to find a better patch, stagnated, stop PSO
            else:
                return True
        else:
            return False

    def _output_repaired_model(self, output_dir, model_repaired):
        # Output
        output_dir = Path(output_dir)
        # Clean directory
        repair_dir = output_dir / "repair"
        if repair_dir.exists():
            shutil.rmtree(repair_dir)
        repair_dir.mkdir()
        # Save model
        model_repaired.save(repair_dir)

    def load_weights(self, weights_dir: Path):
        candidates = []
        with open(weights_dir / "weights.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                candidates.append(row[:3])
        return candidates