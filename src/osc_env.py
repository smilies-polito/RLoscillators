import math
import random
from typing import SupportsFloat, Any
import gymnasium as gym
from gymnasium.core import RenderFrame, ActType, ObsType
import tellurium as te
import libsbml as tesbml
from tellurium.roadrunner.extended_roadrunner import ExtendedRoadRunner as RoadRunner
import matplotlib.pyplot as plt
import numpy as np

PARAM_DISCR = 100.0
PLOT_DELTA = 10


class RRModel(RoadRunner):
    def __init__(self, model_source):
        super().__init__(model_source)
        self.actions = []

    @classmethod
    def load(cls, model_source) -> "RRModel":
        random.seed(666)
        model = None
        try:
            if type(model_source) == str:
                if model_source.endswith('.xml'):
                    try:
                        model = te.loadSBMLModel(model_source)
                        bmSBMLPP = model.getParamPromotedSBML(model)
                        model = te.loadSBMLModel(bmSBMLPP)
                    except Exception as e:
                        print('Could not promote local parameters to global parameters. '
                              'Continuing optimization without parameter promotion.')
                        model = te.loadSBMLModel(model)
                else:  # .ant file or plain str
                    try:
                        model = te.loada(model_source)
                        bmSBML = model.getCurrentSBML()
                        bmSBMLPP = model.getParamPromotedSBML(bmSBML)
                        model = te.loadSBMLModel(bmSBMLPP)
                    except Exception as e:
                        print('Could not promote local parameters to global parameters. '
                              'Continuing optimization without parameter promotion.')
                        model = te.loada(model_source)
            elif isinstance(model_source, RoadRunner):
                try:
                    model = model_source
                    bmSBML = model.getCurrentSBML()
                    bmSBMLPP = model.getParamPromotedSBML(bmSBML)
                    model = te.loadSBMLModel(bmSBMLPP)
                except:
                    print('Could not promote local parameters to global parameters. '
                          'Continuing optimization without parameter promotion.')
                    model = model_source
            else:
                raise RuntimeError('Input not supported: pass an existing roadrunner.RoadRunner instance, '
                                   'Antimony (.ant) file, or an SBML (.xml) file.')

            model.conservedMoietyAnalysis = True
            model.__class__ = cls
            model.param_dict = model.__generate_param_dict()
            model.actions = [model.mul, model.inc, model.dec]
            model.init()
            return model

        except Exception as e:
            print(e)
            raise e

    def __generate_param_dict(self):
        param_list = self._getIndependentFloatingSpeciesIds()
        param_values = list(*self.getFloatingSpeciesConcentrationsNamedArray())

        doc = tesbml.readSBMLFromString(self.getCurrentSBML())
        model = doc.getModel()
        len_list_rules = len(model.getListOfRules())
        assignment_rule_params = []
        for i in range(len_list_rules):
            assignment_rule = model.getRule(i)
            if assignment_rule.getTypeCode() == tesbml.SBML_ASSIGNMENT_RULE:
                assignment_rule_params.append(assignment_rule.getVariable())
        for element in range(len(param_list)):
            for n in assignment_rule_params:
                if (param_list[element].startswith('_CSUM')) or (param_list[element] == n):
                    param_list[element] = 'remove'
                    param_values[element] = 'remove'
        param_list = list(filter(lambda a: a != 'remove', param_list))
        param_values = list(filter(lambda a: a != 'remove', param_values))

        # generate range and delta
        param_ranges = [()] * len(param_values)
        param_deltas = []  # * len(param_values)
        for i in range(len(param_values)):
            if param_values[i] == 0.0:
                param_ranges[i] = (10E-25, 10.0)
            elif param_values[i] <= 10.0 and not param_values[i] < 0.0:
                param_ranges[i] = (param_values[i] / 10.0, 10.0)
            elif param_values[i] > 10.0:
                param_ranges[i] = (param_values[i] / 10.0, 2 * param_values[i])
            param_deltas.append((param_ranges[i][1] - param_ranges[i][0]) / PARAM_DISCR)

        # param_dict = { "param X": {"val": 0.002, "range": (10E-25, 10.0), "delta": range_diff/PARAM_DISCR},
        #                "param Y": {"val": ..., "range": (..., ...), "delta": ...} }
        param_dict = {}
        for k, v, r, d in zip(param_list, param_values, param_ranges, param_deltas):
            param_dict[k] = {"val": v, "range": r, "delta": d}
        print(param_list)

        return param_dict

    def init(self) -> None:
        [self.setValue(f"init({str(k)})", random.uniform(0.0001, 1000)) for k in self.param_dict.keys()]

        self.reset()

        print(self.getFloatingSpeciesConcentrationsNamedArray())

    def __pos_gauss(self, mu, sigma):
        x = random.gauss(mu, sigma)
        return np.abs(x)

    def get_eigenvals(self):
        terminate = False
        self.conservedMoietyAnalysis = True
        eigenvals = None
        no_steady_state = False

        distance = self.steady_state()
        if distance > 10E-6:  # see steady_state() description
            no_steady_state = True
            return np.iinfo(np.int32).max, 0.0, no_steady_state

        try:
            eigenvals = self.getReducedEigenValues()
            return (np.iinfo(np.int32).max, 0.0, no_steady_state) if eigenvals is None else (
            np.real(eigenvals), np.imag(eigenvals), no_steady_state)
        except Exception as e:
            print(f"get_eigenvals 2: {e}")
            no_steady_state = True
            return (np.iinfo(np.int32).max, 0.0, no_steady_state) if eigenvals is None else (
            np.real(eigenvals), np.imag(eigenvals), no_steady_state)

    def steady_state(self, tolerance=10E-15, max_iterations=50):
        self.conservedMoietyAnalysis = True
        self.setSteadyStateSolver('nleq2')
        solver = self.getSteadyStateSolver()
        solver.settings["maximum_iterations"] = max_iterations
        solver.settings["relative_tolerance"] = tolerance
        self.steadyStateSelections = self.getIds()
        distance = np.inf
        try:
            distance = self.steadyState()
        except Exception as e:
            print(f"steady_state: {e}")
        return distance

    def compute_state(self) -> (dict, bool):
        delta = 10E-6
        real_component, imaginary_component, no_steady_state = self.get_eigenvals()

        if no_steady_state or all([i == 0 for i in imaginary_component]):
            return {'real_prod': np.iinfo(np.int32).max, 'imaginary_prod': 0, 'oscillating': 0, 'penalty': -10E3}, False

        num = 1.0
        den = 1.0
        oscillating = 0
        penalty = 0.0
        for n in range(len(real_component)):
            if imaginary_component[n] != 0:
                num = np.abs(num * real_component[n])
                penalty -= np.abs(real_component[n])
            den = den * (1.0 - (0.99 * np.exp(-np.abs(imaginary_component[n]))))
            oscillating += np.log10(np.abs(num / den))  # reward getting close to oscillating
            # https://ocw.mit.edu/courses/2-004-systems-modeling-and-control-ii-fall-2007/2655a03ac5ab9c9b81f135e2903d5d3b_sol09.pdf
            print(f"========= {oscillating} {num} {den}")
        return {'real_prod': num, 'imaginary_prod': den, 'oscillating': oscillating, 'penalty': penalty}, False

    def mul(self, num, var, params, verbose=True) -> None:
        prev = self.getValue(f"init({var})")
        self.setValue(f"init({var})", np.abs(prev * num))  # if np.abs(prev * num) < 10E3 else prev)
        self.reset()
        curr = self.getValue(f"init({var})")
        if verbose:
            print(f"{var}: {prev} -> {curr}")
        return

    def inc(self, num, var, params, verbose=True) -> None:
        prev = self.getValue(f"init({var})")
        self.setValue(f"init({var})", prev + num if prev + num < 1E4 else 1E4)
        self.reset()
        curr = self.getValue(f"init({var})")
        if verbose:
            print(f"{var}: {prev} -> {curr}")
        return

    def dec(self, num, var, params, verbose=True) -> None:
        prev = self.getValue(f"init({var})")
        self.setValue(f"init({var})", prev - num if prev - num > 0 else 0)
        self.reset()
        curr = self.getValue(f"init({var})")
        if verbose:
            print(f"{var}: {prev} -> {curr}")
        return


class OscEnv(gym.Env):
    truncate_at = 1000

    def __init__(self, model_source):
        self.log = ''
        self.verbose = True
        self.elapsed_time = 0
        self.terminated = False
        self.truncated = False
        self.model = None
        self.reward = 0

        try:
            self.model = RRModel.load(model_source)
            self.model.reset()
        except Exception as e:
            raise e

        self.param_dict = self.model.param_dict

        self.num = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 5, 10, 50, 100, 500, 1000]
        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.model.actions), len(self.num), len(self.param_dict.keys())])
        self.observation_names = ['real_prod', 'imaginary_prod', 'oscillating', 'penalty']
        self.observation_names.extend(self.param_dict.keys())
        self.state = {
            'real_prod': 0.0,
            'imaginary_prod': 0.0,
            'oscillating': 0,
            'penalty': 0
        }
        self.state.update({k: 0 for k in self.param_dict.keys()})
        self.observation_space = self.make_observation_space()
        print(self.observation_space)

    def make_observation_space(self) -> gym.spaces.Box:
        lower_obs_bound = {
            'real_prod': 0.0,
            'imaginary_prod': 0.0,
            'oscillating': -np.inf,
            'penalty': -np.inf,
        }
        lower_obs_bound.update({k: 0 for k in self.param_dict.keys()})

        higher_obs_bound = {
            'real_prod': np.inf,
            'imaginary_prod': np.inf,
            'oscillating': np.inf,
            'penalty': 0,
        }
        higher_obs_bound.update({k: np.inf for k in self.param_dict.keys()})

        low = np.array([lower_obs_bound[o] for o in lower_obs_bound.keys()])
        high = np.array([higher_obs_bound[o] for o in higher_obs_bound.keys()])
        shape = (len(higher_obs_bound),)
        return gym.spaces.Box(low, high, shape)

    def reset(self, **kwargs):
        super().reset()
        self.model.init()  # init -> different model at each new episode
        return np.array(self._get_obs(), dtype=np.float32), {}

    def score(self, state):
        reward = state["oscillating"] + state["penalty"]
        print(
            f"step {self.elapsed_time}: reward: {reward} (oscillating: {state['oscillating']},  penalty: {state['penalty']})")
        return reward

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action, num, var = action
        var = list(self.param_dict.keys())[var]
        num = self.num[num]
        print(f"Action {action} {num} on var {var}")
        self.model.actions[action](num, var, self.param_dict[var])
        self.state, self.terminated = self.model.compute_state()
        self.log += f"Action performed: {self.model.actions[action].__name__} on var {var}"

        self.reward = self.score(self.state)

        self.log += str(self.state) + '\n'
        self.elapsed_time += 1

        if self.elapsed_time == OscEnv.truncate_at:
            self.truncated = True
            self.log += 'truncated\n'
        to_return = np.array(self._get_obs(), dtype=np.float32), self.reward, self.terminated, self.truncated, {
            "log": self.log if self.verbose else ""}
        return to_return

    def _get_obs(self):
        self.state, self.terminated = self.model.compute_state()
        st = self.state
        floating_species = list(*self.model.getIndependentFloatingSpeciesConcentrationsNamedArray())
        obs = [math.floor(math.log10(st['real_prod'] + 1)),
               math.floor(math.log10(st['imaginary_prod'] + 1)),
               st['oscillating'],
               st['penalty'],
               *floating_species]
        print(obs)
        return obs

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def plot(self, path, file_name):
        print(f"plot: plotting")
        # self.model.timeCourseSelections = ['time', 'A']
        try:
            self.model.simulate(start=0, end=10, steps=10)
            self.model.plot(title='My plot', xtitle='Time', ytitle='Concentration', dpi=150,
                            savefig=f"{path}/{file_name}_init.png", show=True)
            self.model.reset()

            self.model.simulate(start=0, end=100, steps=1000)
            self.model.plot(title='My plot', xtitle='Time', ytitle='Concentration', dpi=150,
                            savefig=f"{path}/{file_name}.png", show=True)
            self.model.reset()
        except Exception as e:
            print(f"plot exception: {e}")

        self.model.reset()
