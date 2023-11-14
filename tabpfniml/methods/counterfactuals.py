import numpy as np
import pandas as pd
from tabpfniml.methods.interpret import TabPFN_Interpret
from tabpfniml.datasets.datasets import dataset_iml
import gower
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.individual import Individual
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
import random
import time
from typing import Union, List, Optional, Tuple
import os


class Counterfactuals(TabPFN_Interpret):
    """
    Implementation of Multi-Objective Counterfactual Explanations (MOC) using Evolutionary Multi-Objective Algorithms (EMOAs).
    Given an observation of interest (x_factual), they aim to return Counterfactual Explanations (CEs):
    Alternative inputs with minimal changes to x_factual whose predictions equal a desired outcome 
    (that is different to the outcome for x_factual).

    Besides CEs being required to have predictions equal to the desired class (Objective 1: Desired outcome), 
    they should be maximally close to x_factual (Objective 2:  Distance to x_factual), should be sparse in the
    sense that as few features as possible are changed to allow easy interpretations (Objective 3: Sparsity) and
    plausible by adhering to the data manifold (Objective 4: Plausibility).

    The search for CEs can be conducted using Evolutionary Multi-Objective Algorithms (EMOAs) where individuals correspond to
    potential CEs. EMOAs are suitable since they are a member of the class of derivative-free optimization methods for 
    mixed feature spaces and multi-objective problems.

    For faster results, a naive In-Sample Search is also implemented. Thereby, all test-samples are considered as CEs and no EMOA is used.

    Caution: CEs consitute model explanations and should not be confused with the true, underlying data-generating process.
    CEs can also be used to detect model biases.

    Limitations: Only implemented for categorical-labels, not for regression-problems.
    """

    def __init__(self,
                 data: dataset_iml,
                 n_train: int = 1024,
                 n_test: int = 512,
                 N_ensemble_configurations: int = 16,
                 device: str = "cpu",
                 debug: bool = False):
        """
        Initialize a TabPFN-interpretability method by passing rudimentary objects and the general configuration that is constant across all flavors of the interpretability-method.
        For some variables it may be permitted to overwrite them in fit()-methods, although this is not intended in general.

        Args:
            data (dataset_iml, optional): The dataset that TabPFN's behavior shall be explained for. Defaults to dataset_iml.
            n_tran (int, optional): The amount of train-samples to fit TabPFN on. Should not be larger than 1024. Defaults to 512.
            n_test (int, optional): The amount of test-samples to get predictions for. Defaults to 512.
            N_ensemble_configurations (int, optional): The amount of TabPFN forward passes with different augmentations ensembled. Defaults to 16.
            device (str, optional): The device to store tensors and the TabPFN model on. Defaults to "cpu".
            debug (bool, optional): Whether debug mode is activated. This leads to e.g. less train and test samples and can hence tremendously reduce computational cost. Overwrites various other parameters. Defaults to False.
        """
        super().__init__(data=data,
                         n_train=n_train,
                         n_test=n_test,
                         N_ensemble_configurations=N_ensemble_configurations,
                         device=device,
                         debug=debug,
                         standardize_features=False,
                         to_torch_tensor=False,
                         store_gradients=False)

    def evaluate_fitness(self,
                         x_temp_individuals: np.ndarray) -> np.ndarray:
        """
        Helper function to evaluate the fitness of a set of individuals/ CEs w.r.t the objectives.

        Args:
            x_temp_individuals (np.ndarray): The independent features of a set of individuals/ CEs.

        Returns:
            np.ndarray: The fitness values for the set of individuals/ CEs.
        """
        # Objective 1: Desired outcome
        # Treated as hard constraint and hence ensured a-priori/ only candidates are kept that lead to the desired prediction.

        # Objective 2: Distance to x_factual (Gower distance for mixed-feature spaces)
        objective_xspace = gower.gower_matrix(data_x=x_temp_individuals,
                                              data_y=self.x_factual,
                                              cat_features=[(index in self.data.categorical_features_idx) for index in range(self.data.num_features)]).squeeze()
        # self.data.categorical_features_idx).squeeze()

        # Objective 3: Sparsity (Amount of features changed from x_factual)
        objective_sparsity = np.sum(
            x_temp_individuals != self.x_factual, axis=1)

        # Objective 4: Plausibility (Minimal gower distance to an actual observation from the training-data)
        print(x_temp_individuals.shape)
        objective_plausibility = np.min(gower.gower_matrix(
            data_x=x_temp_individuals, data_y=self.X_train), axis=1)

        return np.array([objective_xspace, objective_sparsity, objective_plausibility]).T

    def fit(self,
            test_index_factual: Optional[int] = None,
            x_factual: Optional[dict] = None,
            y_desired: Optional[int] = None,
            in_sample_search: bool = False,
            non_actionable_features: Union[None, List[int], List[str]] = None,
            actionable_features: Union[None, List[int], List[str]] = None,
            population_size: int = 100,
            offsprings_size: int = 50,
            n_iter: int = 10,
            based_on_prior_knowledge: bool = False,
            init_based_on_test_set: bool = False,
            init_poisson_lambda: int = 3,
            print_time: bool = False):
        """
        Uses an Evolutionary Multi-Objective Algorithm (EMOA) to find Counterfactual Explanations (CEs).
        EMOAs evolve a population of individuals (potential CEs) in an interative procedure to approximate the Pareto front of a 
        multi-objective problem. First, they initialize and evaluate a population and then, in an iterative procedure, generate offsprings
        from a set of parents and update the population with the resulting survivors.

        The population must always be of size mu (self.population_size), be sorted (according to NDS) and only consisist of individuals
        with the desired prediction.

        For faster results, a naive In-Sample Search can also be applied. Thereby, all test-samples are considered as CEs and no EMOA is used.

        Sources: 
            -AutoML https://learn.ki-campus.org/courses/automl-luh2021/items/2eQhS3AQNTvLef06Tu8HLT
            -AutoML https://learn.ki-campus.org/courses/automl-luh2021/items/1UVhl6rqF5A04hZOolgvND

        Args:
            test_index_factual(int, optinal): The test-set index of the original/factual observation that CEs shall be computed for. Defaults to 0.
            y_desired (Optional[int], optional): The desired prediction for the CEs. If the dataset has two classes in the target, the contrary class of the predicted class is taken and y_desired is ignored. For multiclass classification (more than two classes in the target), the desired class has to be specified and can not be equal to the actual prediction for x_factual. Defaults to None.
            in_sample_search (bool, optional): Whether to apply in-sample search instead of EMOA. The resulting CEs are expected to have lower quality, however they can be obtained ways faster. Ignores non_actionable_features, population_size, offsprings_size, n_iter, based_on_prior_knowledge, init_based_on_test_set, init_poisson_lambda, init_poisson_lambda, print_time. Defaults to False.
            non_actionable_features (Union[None, List[int], List[str]], optional): If specified, a list of feature indices or names of features whose values should not be changed from x_factual for the generation of CEs. Ignored if init_based_on_test_set= True. Only actionable- or non-actionable features can be specified. Defaults to None.
            actionable_features (Union[None, List[int], List[str]], optional): If specified, a list of feature indices or names of features whose values can be changed from x_factual for the generation of CEs. Ignored if init_based_on_test_set= True. Only actionable- or non-actionable features can be specified. Defaults to None.
            population_size (int, optional): The amount of individuals/ potential CEs in the population (mu). Must be at least twice as large as offsprings_size in order to perform crossover when generating offsprings. Defaults to 100.
            offsprings_size (int, optional): The amount of offsprings to generate in each iteration of EMOA (lambda). Defaults to 50.
            n_iter (int, optional): The amount of iterations in the iterative optimization-procedure (EMOA). Defaults to 10.
            based_on_prior_knowledge (bool, optional): Whether to compute mutations based on prior knowledge (feature-gradients as indicator of how sensitive the model is to them). NOT IMPLEMENTED YET. Defaults to False.
            init_based_on_test_set (bool, optional): Whether to initialize the population using the test-set or by mutating around x_factual. Mutating around x_factual is expected to yield higher sparsity (Objective 3) and lower distance to x_factual (Objective 2), however this may come at the cost of lower plausibility (Objective 4). If set to True, non_actionable_features and based_on_prior_knowledge are ignored. For small test sets (e.g. in debug mode), initialization based on the test set may lead to an empty initial population. However, the mutation sometimes this takes very long in order to yield a suitable initial population. Defaults to False.
            init_poisson_lambda (int, optional): The initial lambda-parameter used when sampling from a Poisson-distribution the amount of features to be changed when mutating x_factual. Defaults to 3.
            print_time (bool, optional): Whether to print the runtime of several steps of the EMOA. Defaults to False.

        Raises:
            Exception: If init_based_on_test_set= True although non_actionable_features != None.
            Exception: If non_actionable_features has been specified, however in a non-valid format.
            Exception: If the specified y_desired does not exist. 
            Exception: If the specified y_desired has not been specified in multiclass-classification.
        """

        def sample_num_changed_features(poisson_lambda: float) -> int:
            """
            Helper function that samples the amount of features to be changed from x_factual when mutating around x_factual.
            This is used to ensure sparsity (Objective 3).

            Args:
                poisson_lambda (float): The lambda-parameter used when sampling from a Poisson-distribution the amount of features to be changed from x_factual.

            Returns:
                int: The amount of features to be changed.
            """
            num_changed_features = np.random.poisson(poisson_lambda, 1)[0]

            num_changed_features = (num_changed_features if (num_changed_features < len(
                self.actionable_features)) else len(self.actionable_features))
            num_changed_features = (num_changed_features if (
                num_changed_features > 0) else 1)

            return num_changed_features

        def non_dominated_sorting(x_temp_individuals: np.ndarray,
                                  pred_temp_individuals: np.ndarray,
                                  fitness_temp_individuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Applies Non-dominated Sorting to a set of individuals/ CEs and returns the resulting set that is reduced to the
            population size mu and sorted. In detail, NDS partitions individuals in the objective space into Pareto fronts to 
            rank them under multiple objectives and Crowding Distance Assignment is utilized as tie breaker.

            Args:
                x_temp_individuals (np.ndarray): The independent features of a set of individuals/ CEs.
                pred_temp_individuals (np.ndarray): The model predictions of the same set of individuals/ CEs.
                fitness_temp_individuals (np.ndarray): The fitness values of the same set of individuals/ CEs.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The independent features, model predictions and fitness values of the resulting individuals after applying NDS.
            """
            # Compute Pareto frontss to rank individuals under multiple objectives
            nds = NonDominatedSorting()
            pareto_fronts = nds.do(fitness_temp_individuals)

            # List with all survivors that will be kept
            survivors_ids = []

            # Keep track of amount of survivors that are remaining until mu survivors are reached
            remaining_survivors = self.population_size

            # Add individuals in Pareto fronts sequentially to the survivors until the next Pareto front does not fully fit anymore
            pareto_front_index = 0
            while (len(pareto_fronts[pareto_front_index]) <= remaining_survivors) and (len(pareto_fronts[pareto_front_index]) > 0):
                survivors_ids += list(pareto_fronts[pareto_front_index])
                remaining_survivors -= len(pareto_fronts[pareto_front_index])
                pareto_front_index += 1
                if (pareto_front_index + 1) > len(pareto_fronts):
                    break

            # Add individuals from the next Pareto front using Crowding Distance Assignment
            # Only if remaining survivors are left and individuals are left
            try:
                if remaining_survivors > 0 and len(pareto_fronts[pareto_front_index]) > 0:
                    crowding_distance = calc_crowding_distance(
                        fitness_temp_individuals[pareto_fronts[pareto_front_index], :])
                    survivors_ids += list(pareto_fronts[pareto_front_index]
                                          [crowding_distance.argsort()[::-1][:remaining_survivors]])
            except:
                pass

            return x_temp_individuals[survivors_ids, :], pred_temp_individuals[survivors_ids], fitness_temp_individuals[survivors_ids, :]

        def mutate_x(x_individual: np.ndarray,
                     var_factor: float = 1,
                     poisson_lambda: int = 3,
                     n_mutations: int = 1,
                     based_on_prior_knowledge: bool = False) -> np.ndarray:
            """
            Locally change an individual/ CE using Gaussian mutation for continuous variables and a random different level 
            for categorical variables. Only actionable features are changed.

            Args:
                x_individual (np.ndarray): The independent features of a single individual/ CE.
                var_factor (float, optional): The factor to use when enhancing the variance of the Gaussian mutation in each step. Defaults to 1.
                n_mutations (int, optional): The amount of mutations to generate from the individual/ CE. Defaults to 1.
                based_on_prior_knowledge (bool, optional): Whether to compute mutations based on prior knowledge (feature-gradients as indicator of how sensitive the model is to them). NOT IMPLEMENTED YET. Defaults to False.

            Raises:
                Exception: Mutation based on prior knowledge is not implemented yet.

            Returns:
                np.ndarrays: The mutated individual/ CE.
            """

            x_mutations = []
            for _ in range(n_mutations):
                # Sample the amount of features to be mutated. Do not mutate all features to maintain low distance to x_factual (Objective 2)
                # and to maintain sparsity (Objective 3).
                num_changed_features = sample_num_changed_features(
                    poisson_lambda)

                if based_on_prior_knowledge:
                    # Sample the features to be changes using prior knowledge (gradients, e.g. those with higher gradients have higher probability to be mutated)
                    # TODO (should also set changed_features)
                    raise Exception("Not implemented yet.")

                else:
                    # Only mutate actionable features
                    changed_features = random.sample(self.actionable_features,
                                                     num_changed_features)  # Samples without replacement.

                x_mutation = x_individual.copy()
                for feature in changed_features:
                    if feature in self.data.continuous_features:
                        # For continuous features utilize Gaussian mutation
                        # print("Feature value of feature {} before mutation: {}".format(feature, offspring[feature]))
                        x_mutation_temp_feature = x_mutation[feature] + np.random.normal(loc=0,
                                                                                         scale=self.sd_cont_feats[
                                                                                             feature] * var_factor,
                                                                                         size=(1,))

                        # Ensure that mutation is not greater than maximum value and not smaller than minimum value of feature
                        if (x_mutation_temp_feature >= self.min_vals[feature] and x_mutation_temp_feature <= self.max_vals[feature]):
                            x_mutation[feature] = x_mutation_temp_feature
                        elif x_mutation_temp_feature < self.min_vals[feature]:
                            x_mutation[feature] = self.min_vals[feature]
                        else:
                            x_mutation[feature] = self.max_vals[feature]
                        # print("Feature value of feature {} after mutation: {}".format(feature, offspring[feature]))

                    else:
                        # For categorical features sample randomly from all dfferent feature levels
                        try:
                            x_mutation[feature] = np.random.choice(
                                (self.unique_vals_per_feat[feature][self.unique_vals_per_feat[feature] != x_mutation[feature]]), 1)
                        except:
                            # If no other feature level exists in the data, maintain the value from x
                            pass
                x_mutations.append(x_mutation)

            return np.array(x_mutations)

        def do_crossover(x_parent_a: np.ndarray,
                         x_parent_b: np.ndarray) -> np.ndarray:
            """
            Combine two parent individuals/ CE into one offspring using crosssover. For continuous features 
            SimulatedBinaryCrossover (SBX) is applied and Uniform crossover for categorical features.
            Only actionable features are changed.

            Args:
                x_parent_a (np.ndarray): The independent features of the first parent, a single individual/ CE.
                x_parent_b (np.ndarray): The independent features of the second parent, a single individual/ CE.

            Returns:
                np.ndarray: The resulting offspring.
            """
            x_offspring = self.x_factual.copy().squeeze()
            for feature in self.actionable_features:
                if feature in self.data.continuous_features:
                    # For continuous features utilize SimulatedBinaryCrossover (SBX)
                    problem = Problem(n_var=1,
                                      xl=self.min_vals[feature],
                                      xu=self.max_vals[feature])
                    x_parent_a_feat = Individual(X=np.expand_dims(
                        x_parent_a[feature].copy(), axis=0))
                    x_parent_b_feat = Individual(X=np.expand_dims(
                        x_parent_b[feature].copy(), axis=0))
                    x_parent_feats = [[x_parent_a_feat, x_parent_b_feat]]
                    off = SBX(prob=1.0, prob_var=1.0, eta=0.1).do(
                        problem, x_parent_feats)
                    # Returns twice the amount of offsprings - one biased towards x_parent_a and one towards x_parent_b
                    x_offspring[feature] = off.get("X")[random.randrange(2)]

                else:
                    # For categorical features use Uniform crossover
                    x_offspring[feature] = (x_parent_a[feature].copy() if (
                        random.randrange(2) == 0) else x_parent_b[feature].copy())

            return np.array([x_offspring])

        def initialize_population(based_on_test_set: bool = False,
                                  based_on_prior_knowledge: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Generates and evaluates an initial population consisting of individuals/ potential CEs.
            One can either initialize the population using the test-set or mutating around x_factual. If mutating around x_factual,
            for continuous features Gaussian mutation is utilized and a random different level for categorical variables.

            It ensures that the initial populatiion is of size self.population_size (mu) and only contains individuals with the
            desired prediction. Furthermore, the initial population is sorted according to NDS.

            Args:
                based_on_test_set (bool, optional): Whether to initialize the population using the test-set. Mutating around x_factual is expected to yield higher sparsity (Objective 3) and distance to x_factual (Objective 2), however this may come at the cost of lower plausibility (Objective 4). If set to True, non_actionable_features and based_on_prior_knowledge are ignored. Defaults to False.
                based_on_prior_knowledge (bool, optional): Whether to compute mutations based on prior knowledge (feature-gradients as indicator of how sensitive the model is to them). NOT IMPLEMENTED YET. Defaults to False.


            Raises:
                Exception: If based_on_test_set= True and no observation with the desired predictiion is in the  test set.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The independent features, model predictions and fitness values of the initial population sorted by NDS.
            """
            start = time.time()
            if based_on_test_set:
                # Initialiize the population using the test-set
                # Caution: Non-actionable features ignored in this setting

                x_initial_candidates = self.X_test.copy()
                pred_initial_candidates = self.classifier.predict(
                    x_initial_candidates)

                # Filter observations with desired prediction
                desired_initial_candidates = (
                    pred_initial_candidates == self.y_desired)
                if desired_initial_candidates.sum() == 0:
                    raise Exception("No counterfactual with the desired prediction was found in test set to initialize the population with. \
                                    As an alternative you could iniitialize the population by mutating around x_factual. \
                                    Therefore set init_based_on_test_set= False in the fit-method.")
                x_initial_candidates = x_initial_candidates[desired_initial_candidates, :]
                pred_initial_candidates = pred_initial_candidates[desired_initial_candidates]

                # Make sure initial population is large enough
                # If not, duplicate test data
                if x_initial_candidates.shape[0] < self.population_size:
                    while x_initial_candidates.shape[0] < self.population_size:
                        x_initial_candidates = np.concatenate(
                            [x_initial_candidates, x_initial_candidates], axis=0)
                        pred_initial_candidates = np.concatenate(
                            [pred_initial_candidates, pred_initial_candidates], axis=0)
                    x_initial_candidates = x_initial_candidates[:self.population_size, :]
                    pred_initial_candidates = pred_initial_candidates[:self.population_size]

            else:
                # Initialize the population mutating around x_factual
                # For categorical features it is sampled randomly from all different feature levels

                init_pop_size_factor = 50
                increase_factor_to_iter = 10
                # Alternative: Inverse share of mutations with desired pred

                start = time.time()
                init_iter = 0

                x_initial_candidates = np.empty(
                    shape=(0, self.data.num_features))
                pred_initial_candidates = np.empty(shape=(0,))

                # Obtain at least self.population_size many mutations with the desired predictiion
                while x_initial_candidates.shape[0] < self.population_size:
                    # For continuous features Gaussian mutation is utilized and  a random different level for categorical variables
                    # In order to ensure that the desired prediction is obtained, the Gaussian mutation variance and amount of mutations is
                    # enhanced until enough mutations with the desired prediction are obtained.
                    x_initial_candidates_temp = mutate_x(x_individual=self.x_factual.copy().squeeze(),
                                                         n_mutations=self.population_size *
                                                         (init_pop_size_factor + init_iter *
                                                          increase_factor_to_iter),
                                                         var_factor=(
                                                             0.05 * (init_iter + 1)),
                                                         poisson_lambda=(
                                                             self.init_poisson_lambda + init_iter),
                                                         based_on_prior_knowledge=based_on_prior_knowledge)
                    pred_initial_candidates_temp = self.classifier.predict(
                        x_initial_candidates_temp)

                    # Filter mutations with desired prediction
                    desired_initial_candidates_temp = (
                        pred_initial_candidates_temp == self.y_desired)
                    x_initial_candidates_temp = x_initial_candidates_temp[
                        desired_initial_candidates_temp, :]
                    pred_initial_candidates_temp = pred_initial_candidates_temp[
                        desired_initial_candidates_temp]

                    # Add to initial_population
                    x_initial_candidates = np.concatenate(
                        [x_initial_candidates, x_initial_candidates_temp], axis=0)
                    pred_initial_candidates = np.concatenate(
                        [pred_initial_candidates, pred_initial_candidates_temp], axis=0)
                    # print("With {} proposed samples, sd of {} and poisson lambda of {}, the initial population grew to {}".format((init_pop_size_factor+iter*increase_factor_to_iter)*self.population_size, 0.1*(iter+1), 3+iter, x_initial_population.shape[0]))

                    init_iter += 1

            fitness_initial_candidates = self.evaluate_fitness(
                x_initial_candidates)

            # Reduce to population size mu individuals via NDS
            x_initial_population, pred_initial_population, fitness_initial_population = non_dominated_sorting(
                x_initial_candidates, pred_initial_candidates, fitness_initial_candidates)

            if self.print_time:
                print("Time to initialize and evaluate population: {}".format(
                    time.time()-start))

            return x_initial_population, pred_initial_population, fitness_initial_population

        def select_parents(x_population: np.ndarray) -> np.ndarray:
            """
            Select parents from the current population using tournament selection.
            In order to apply crossover to parents and obtain self.offsprings_size (lambda) offsprings in the next step, 2* self.offsprings_size (2* lambda) parents are selected.

            Args:
                x_population (np.ndarray): The independent features of the current population/ CEs of size self.population_size (mu). Assumes that the population is sorted according to the objectives.

            Returns:
                np.ndarray: A set of 2* self.offsprings_size (2* lambda) parents/ CEs.
            """
            parent_size = 2 * self.offsprings_size

            # Assumes that x_population is sorted. (initialize_population and select_survivors both apply NDS and thereby sort)
            rank_population = np.arange(x_population.shape[0])
            x_parents = np.zeros(shape=(0, self.data.num_features))

            for _ in range(parent_size):
                # The higher tournament_fraction, the more distinct parents are generated. For 1, the sampled parent would always be the same.
                tournament_fraction = 4

                tournament_group = random.sample(range(x_population.shape[0]), max(
                    1, int(self.population_size/tournament_fraction)))  # No replacement
                tournament_winner = rank_population[tournament_group].min()
                x_parents = np.concatenate([x_parents, np.expand_dims(
                    x_population[tournament_winner, :], axis=0)], axis=0)

            shuffle = random.sample(range(parent_size), parent_size)
            x_parents = x_parents[shuffle, :]

            return x_parents

        def generate_offsprings(x_parents: np.ndarray) -> np.ndarray:
            """
            Given a set of parents, offsprings are generated using crossover (comnbining two parents) and mutation (locally chaning an individual).

            Args:
                x_parents (np.ndarray): The independent features of 2* self.offsprings_size (2* lambda) parents.


            Returns:
                np.ndarray: A set of self.offsprings_size (lambda) or less offsprings/ CEs.
            """

            start = time.time()
            x_offsprings = np.zeros(
                (self.offsprings_size, self.data.num_features))

            for i in range(self.offsprings_size):
                # Select 2 parents
                x_parent_a = x_parents[2*i, :].copy()
                x_parent_b = x_parents[2*i+1, :].copy()

                # Step 1: Apply crossover (SBX for numerical, Uniform crossover for categorical)
                x_offspring = do_crossover(x_parent_a=x_parent_a,
                                           x_parent_b=x_parent_b)

                # Step 2: Apply mutation (Gaussian mutation for numerical, sampling from levels for categorical)
                x_offspring = mutate_x(x_individual=x_offspring.squeeze(),
                                       n_mutations=1,
                                       var_factor=random.uniform(a=0.1, b=1),
                                       poisson_lambda=self.init_poisson_lambda,
                                       based_on_prior_knowledge=self.based_on_prior_knowledge)[0]

                x_offsprings[i, :] = x_offspring

            if self.print_time:
                print("Time for crossover and mutation: {}".format(
                    time.time()-start))

            start = time.time()
            preds_offsprings = self.classifier.predict(x_offsprings)
            fitness_offsprings = self.evaluate_fitness(x_offsprings)
            if self.print_time:
                print("Time to evaluate offsprings: {}".format(time.time()-start))

            # Filter offsprings that do not lead to y_desired
            # Therefore less then self.offsprings_size offsprings may be returned
            desired_offsprings = (preds_offsprings == self.y_desired)
            x_offsprings = x_offsprings[desired_offsprings, :]
            preds_offsprings = preds_offsprings[desired_offsprings]
            fitness_offsprings = fitness_offsprings[desired_offsprings, :]

            return x_offsprings, preds_offsprings, fitness_offsprings

        def select_survivors(x_population: np.ndarray,
                             x_offsprings: np.ndarray,
                             preds_population: np.ndarray,
                             preds_offsprings: np.ndarray,
                             fitness_population: np.ndarray,
                             fitness_offsprings: np.ndarray) -> np.ndarray:
            """
            Select from self.population_size+ self.offspring_size (mu+ lambda) population individuals and offsprings the best self.population_size (mu) survivors/ CEs according to NDS.

            Args:
                x_population (np.ndarray): The independent features of the current population of size self.population_size (mu).
                x_offsprings (np.ndarray): The independent features of the offsprings of size self.offspring_size (lambda).
                preds_population (np.ndarray): The model predictions of the current population of size self.population_size (mu).
                preds_offsprings (np.ndarray): The model predictions of the offsprings of size self.offspring_size (lambda).
                fitness_population (np.ndarray): The fitness values of the current population of size self.population_size (mu).
                fitness_offsprings (np.ndarray): The fitness values of the offsprings of size self.offspring_size (lambda).

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The independent features, model predictions and fitness values of the survivors, sorted by NDS and of size self.population_size (mu).
            """
            # It is ensured that only survivors with the desired predictions are generated, since the population and the offsprings only consist of individuals with the desired prediction
            x_population, preds_population, fitness_population = non_dominated_sorting(x_temp_individuals=np.concatenate([x_population, x_offsprings], axis=0),
                                                                                       pred_temp_individuals=np.concatenate(
                                                                                           [preds_population, preds_offsprings], axis=0),
                                                                                       fitness_temp_individuals=np.concatenate([fitness_population, fitness_offsprings], axis=0))

            return x_population, preds_population, fitness_population

        def do_in_sample_search() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Perform a naive In-Sample Search instead of EMOA. 
            The resulting CEs are expected to have lower quality, however they can be obtained ways faster.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]: The independent features, model predictions and fitness values of the survivors, sorted by NDS and of size at most self.population_size (mu).
            """
            x_candidates = self.X_test.copy()
            pred_candidates = self.classifier.predict(x_candidates)

            # Filter observations with desired prediction
            desired_candidates = (pred_candidates == self.y_desired)
            if desired_candidates.sum() == 0:
                raise Exception("No CEs with the desired prediction was found in test set to perform in-sample search. \
                                As an alternative you could use EMOA by setting in_sample_search= False.")
            x_candidates = x_candidates[desired_candidates, :]
            pred_candidates = pred_candidates[desired_candidates]
            fitness_candidates = self.evaluate_fitness(x_candidates)

            # Apply NDS to sort the candidates
            x_population, preds_population, fitness_population = non_dominated_sorting(x_temp_individuals=x_candidates,
                                                                                       pred_temp_individuals=pred_candidates,
                                                                                       fitness_temp_individuals=fitness_candidates)

            return x_population, preds_population, fitness_population

        self.in_sample_search = in_sample_search
        self.test_index_factual = test_index_factual
        self.n_iter = n_iter
        self.population_size = population_size
        self.offsprings_size = offsprings_size
        self.based_on_prior_knowledge = based_on_prior_knowledge
        self.init_based_on_test_set = init_based_on_test_set
        self.init_poisson_lambda = init_poisson_lambda
        self.print_time = print_time

        # The population size (mu) needs to be twice as large as the amount of offsprings (lambda), since each offspring requires 2 parents for crossover.
        assert self.population_size >= (2 * self.offsprings_size)

        #If non_actionable_features or actionable_features are specified, ensure that they are valid lists of ints
        def feature_list_preprocessing(feature_list: List) -> List: 
            """Helper method"""
            is_all_strings = all(isinstance(item, str)
                                    for item in feature_list)
            is_all_ints = all(isinstance(item, int)
                                for item in feature_list)

            if is_all_strings:
                try:
                    feature_list = [self.data.feature_name_to_index(
                        feature_name) for feature_name in feature_list]
                except:
                    raise Exception(
                        "A feature has been specified as a string that does not occur in the dataset.")
            elif is_all_ints:
                for feature_index in feature_list:
                    assert (
                        (feature_index) < self.data.num_features), "Feature {} does not occur in the dataset".format(str(feature_index))
            else:
                raise Exception(
                    "The features have to be specified either as a valid list of strings (of feature names) or as a valid list of ints (of feature indices).")
            
            return feature_list
        
        if non_actionable_features is not None and actionable_features is not None:
            raise Warning(
                "non_actionable_features and actionable_features cannot be specified at the same time.")
        
        elif non_actionable_features is not None:
            self.non_actionable_features= feature_list_preprocessing(non_actionable_features)
            self.actionable_features= list(
                set(list(range(self.data.num_features)))-set(self.non_actionable_features))

        elif actionable_features is not None:
            self.actionable_features= feature_list_preprocessing(actionable_features)
            self.non_actionable_features = list(
                set(list(set(np.arange(self.data.num_features))))-set(self.actionable_features))
        else:
            self.non_actionable_features = None
            self.actionable_features = list(set(np.arange(self.data.num_features)))

        if self.non_actionable_features is not None:
            if self.init_based_on_test_set:
                raise Exception("Basing the initial population on the test set does not work if non-actionable features are specified. \
                                You can set non_actionable_features= None or init_based_on_test_set= False to solve this issue.")

        self.classifier.fit(self.X_train, self.y_train)

        # The original/factual observation that CEs shall be computed for.
        if x_factual is not None and test_index_factual is not None:
            raise Warning(
                "x_factual and test_index_factual cannot be passed both at the same time.")
        elif x_factual is not None:
            x_factual_df = pd.DataFrame(columns=self.data.feature_names)
            x_factual_df = pd.concat([x_factual_df, pd.DataFrame(
                x_factual, index=[0])], ignore_index=True)
            self.x_factual = x_factual_df.to_numpy().astype(float)
        elif test_index_factual is not None:
            self.x_factual = self.X_test[self.test_index_factual, :].copy().reshape(
                1, -1)
        else:
            raise Warning("x_factual or test_index_factual have to be passed.")
        # The model predicton for x_factual.
        self.pred_factual = self.classifier.predict(self.x_factual)[
            0]  # pred interest

        # Infer the desired class for the CEs.
        y_labels_wo_predicted = list(range(self.data.y_classes))
        y_labels_wo_predicted.remove(self.pred_factual)
        if len(y_labels_wo_predicted) == 1:
            # If the dataset has two classes in the target, the contrary class of the predicted class is taken.
            self.y_desired = y_labels_wo_predicted[0]
        else:
            # Otherwise the desired class is taken if possible and if specified.
            if y_desired is not None:
                if y_desired in y_labels_wo_predicted:
                    self.y_desired = y_desired
                else:
                    raise Exception(
                        "The desired class (y_desired) does not exist for the dataset.")
            else:
                raise Exception(
                    "The desired class has to be specified for multiclass-classification.")

        if self.debug:
            self.population_size = 30
            self.offsprings_size = 15
            self.n_iter = 3

        # Get minimum and maximum value per feature in the training data to ensure that all individuals are always within those bounds.
        # TODO: Outsource to dataset-object
        self.min_vals = np.min(self.X_train.copy(), axis=0)
        self.max_vals = np.max(self.X_train.copy(), axis=0)

        # TODO: Outsource to dataset-object
        self.unique_vals_per_feat = [np.unique(
            self.X_train[:, col].copy()) for col in range(self.data.num_features)]
        # self.levels_per_feat = np.array([len(unique_vals) for unique_vals in self.unique_vals_per_feat])
        self.sd_cont_feats = [np.sqrt(np.var(self.X_train[:, col].copy())) if (
            col in self.data.continuous_features) else 0 for col in range(self.data.num_features)]

        # Option 1: EMOA
        if not self.in_sample_search:
            # Step 1 & 2: Initialize the population and evaluate the initial populaton
            self.x_population, self.preds_population, self.fitness_population = initialize_population(based_on_test_set=self.init_based_on_test_set,
                                                                                                      based_on_prior_knowledge=self.based_on_prior_knowledge)

            # Start iterative procedure
            for emoa_iter in range(self.n_iter):
                # Step 3: Parent selection via Tournament selection
                x_parents = select_parents(
                    x_population=self.x_population.copy())

                # Step 4: Generate offsprings and evaluate them
                x_offsprings, preds_offsprings, fitness_offsprings = generate_offsprings(
                    x_parents=x_parents)

                # Step 5: Select survivors from population and offsprings
                self.x_population, self.preds_population, self.fitness_population = select_survivors(x_population=self.x_population.copy(),
                                                                                                     x_offsprings=x_offsprings,
                                                                                                     preds_population=self.preds_population.copy(),
                                                                                                     preds_offsprings=preds_offsprings,
                                                                                                     fitness_population=self.fitness_population.copy(),
                                                                                                     fitness_offsprings=fitness_offsprings)

                # Check that only actionable features have been changed
                if not self.init_based_on_test_set and self.non_actionable_features:
                    np.testing.assert_array_equal(x=self.x_factual[:, self.non_actionable_features],
                                                  y=np.unique(self.x_population[:, self.non_actionable_features], axis=0))

        # Option 2: Naive In-Sample Search
        else:
            self.x_population, self.preds_population, self.fitness_population = do_in_sample_search()

        # Verify that x_population and fitness_population elements and order stil match, by computing fitness-scores from x_population again
        for i in range(self.x_population.shape[0]):
            assert (int(self.data.num_features -
                    (self.x_population[i] == self.x_factual).sum()) == self.fitness_population[i][1])

    def get_counterfactuals(self,
                            first_n: int = 10,
                            save_to_path: Optional[str] = None) -> pd.DataFrame:
        """
        Returns Counterfactual Explanations (CEs) generated in the fit()-function. They represent alternative inputs with minimal 
        changes to x_factual whose predictions equal a desired outcome. They should be maximally close to x_factual (Objective 2:  
        Distance to x_factual), be sparse in the sense that as few features as possible are changed to allow easy interpretations 
        (Objective 3: Sparsity) and plausible by adhering to the data manifold (Objective 4: Plausibility).

        Args:
            first_n (int, optional): The amount of CEs to return. Defaults to 10.
            save_to_path (Optional[str], optional): If provided, save the dataframe to the specified path. Should end with '.csv'. Defaults to None.

        Raises:
            ValueError: If the specified path for saving the dataframe does not work.

        Returns:
            pd.DataFrame: A dataframe with x_factual as first row and the best CEs as further rows. The columns provide the fitness values of the CEs (where every objective should be minimized) and the independent features of the CEs.
        """

        # Reduce to unique set for presentation. May have duplicates otherwise, especially when fitting with self.init_based_on_test_set= True.
        _, index = np.unique(self.x_population.copy(),
                             axis=0, return_index=True)
        index.sort()

        x_population = self.x_population.copy()[index, :]
        preds_population = self.preds_population.copy()[index]
        fitness_population = self.fitness_population.copy()[index, :]

        # Verify that x_population and fitness_population elements and order stil match, by computing fitness-scores from x_population again
        for i in range(x_population.shape[0]):
            assert (int(self.data.num_features -
                    (x_population[i] == self.x_factual).sum()) == fitness_population[i][1])

        if self.non_actionable_features is not None:
            results = np.concatenate([np.expand_dims(preds_population, axis=1),
                                      fitness_population,
                                      x_population[:,
                                                   self.non_actionable_features],
                                      x_population[:, self.actionable_features]], axis=1)
            results = np.concatenate([np.concatenate([np.array([[self.pred_factual]]),
                                                      np.repeat(np.expand_dims(
                                                          np.array([0]), axis=1), 3, 1),
                                                      self.x_factual[:,
                                                                     self.non_actionable_features],
                                                      self.x_factual[:, self.actionable_features]], axis=1),
                                      results], axis=0)
            results = pd.DataFrame(results,
                                   columns=pd.MultiIndex.from_tuples([("Objectives", metric) for metric in ["Prediction", "Closeness", "Sparsity", "Plausibility"]] +
                                                                     [("Non-actionable Features", self.data.feature_names[feat_ind]) for feat_ind in self.non_actionable_features] +
                                                                     [("Actionable Features", self.data.feature_names[feat_ind]) for feat_ind in self.actionable_features]),
                                   index=["x_factual"] + ["CE " + str(j+1) for j in range(fitness_population.shape[0])])

        else:
            results = np.concatenate([np.expand_dims(preds_population, axis=1),
                                      fitness_population,
                                      x_population], axis=1)
            results = np.concatenate([np.concatenate([np.array([[self.pred_factual]]),
                                                      np.repeat(np.expand_dims(
                                                          np.array([0]), axis=1), 3, 1),
                                                      self.x_factual], axis=1),
                                      results], axis=0)

            results = pd.DataFrame(results,
                                   columns=pd.MultiIndex.from_tuples([("Objectives", metric) for metric in ["Prediction", "Closeness", "Sparsity", "Plausibility"]] +
                                                                     [("Features", feat_name) for feat_name in self.data.feature_names]),
                                   index=["x_factual"] + ["CE " + str(j+1) for j in range(fitness_population.shape[0])])

        results = results.iloc[:(first_n+1), :]

        if save_to_path is not None:
            try:
                if not os.path.exists(os.path.dirname(save_to_path)):
                    os.makedirs(os.path.dirname(save_to_path))
                results.to_csv(save_to_path)
            except:
                raise ValueError(
                    "The specified path does not work. The path should end with '.csv'.")
        return results
