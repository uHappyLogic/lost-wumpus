import copy as cp
import random

from action import Action


def weighted_choice(choices):

    total = 0
    for choice in choices:
        total += choices[choice][0]

    r = random.uniform(0, total)
    upto = 0
    for choice in choices:
        if upto + choices[choice][0] >= r:
            return choice
        upto += choices[choice][0]


class Agent:

    def __init__(self, p, pj, pn, height, width, areaMap):

        self.times_moved = 0
        self.direction = Action.LEFT

        self.p = p
        self.p_d = (1 - p) / float(4)
        self.pj = pj
        self.pn = pn
        self.height = height
        self.width = width
        self.map = [[str(x) for x in line] for line in areaMap]

        self.indicators = []

        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] in ['J', 'W']:
                    self.indicators.append([i, j])

        self.find_exit_coord()

        self.last_move = '.'
        self.opposite_move = {
            Action.DOWN: Action.UP,
            Action.UP: Action.DOWN,
            Action.RIGHT: Action.LEFT,
            Action.LEFT: Action.RIGHT,
            '.': '.'
        }

        self.conditional_probs = {
            (True, 'J'): self.pj,
            (True, '.'): self.pn,
            (False, 'J'): (1 - self.pj),
            (False, '.'): (1 - self.pn)
        }

        self.fields_count = { ch: sum([line.count(ch) for line in self.map]) for ch in ['.', 'J']}
        self.field_probs = { ch: (self.fields_count[ch] / float(width * height)) for ch in ['.', 'J']}

        self.result_probs = {
            True: (self.pj + self.pn) / float(2),
            False: (1 - self.pj) + (1 - self.pn) / float(2),
        }

        self.hist = []
        for y in range(self.height):
            self.hist.append([])
            for x in range(self.width):
                self.hist[y].append(1.)

        self.hist[self.exit_coords[0]][self.exit_coords[1]] = 0

        self.prepare_directions()

        return

    def sense(self, sensor):
        self.apply_sense_on_histo(sensor)
        self.normalize_hist()

    def move(self):

        current_decision = self.calculate_direction()

        self.move_histo(current_decision)

        self.normalize_hist()

        self.last_move = current_decision

        self.times_moved += 1
        return current_decision

    def histogram(self):
        return self.hist

    # =============================================================================================
    # =======================================IMPL==================================================
    # =============================================================================================

    def prepare_directions(self):
        self.directions = [
            [
                [] for j in range(self.width)
            ] for i in range(self.height)
        ]

        for i in range(self.height):
            for j in range(self.width):

                if [i, j] == self.exit_coords:
                    self.directions[i][j] = ['W']
                    continue

                direction = []

                wg = lambda decision, distance, t: [decision, (t - distance) / float(t)]

                v_d_inner = self.exit_coords[1] - j

                if v_d_inner < 0:
                    outer_d = (self.width - j + self.exit_coords[1])
                    v_d_inner = abs(v_d_inner)

                    if v_d_inner < outer_d:
                        direction.append(wg(Action.LEFT, v_d_inner, self.width))
                    else:
                        direction.append(wg(Action.RIGHT, outer_d, self.width))

                elif v_d_inner > 0:
                    outer_d = (j + self.width - self.exit_coords[1])

                    if v_d_inner < outer_d:
                        direction.append(wg(Action.RIGHT, v_d_inner, self.width))
                    else:
                        direction.append(wg(Action.LEFT, outer_d, self.width))

                h_d_inner = self.exit_coords[0] - i

                if h_d_inner < 0:
                    h_d_inner = abs(h_d_inner)
                    outer_d = (self.height - i + self.exit_coords[0])

                    if h_d_inner < outer_d:
                        direction.append(wg(Action.UP, h_d_inner, self.height))
                    else:
                        direction.append(wg(Action.DOWN, outer_d, self.height))
                elif h_d_inner > 0:
                    outer_d = (i + self.height - self.exit_coords[0])

                    if h_d_inner < outer_d:
                        direction.append(wg(Action.DOWN, h_d_inner, self.height))
                    else:
                        direction.append(wg(Action.UP, outer_d, self.height))

                self.directions[i][j] = direction

    def calculate_direction(self):
        result = {act: [0, 0] for act in [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]}

        for i in range(self.height):
            for j in range(self.width):
                if [i, j] == self.exit_coords:
                    continue

                if self.hist[i][j] <= 0.9:
                    continue

                for move, weight in self.directions[i][j]:
                    if self.is_opposite_to_last(move):
                        continue
                    result[move][0] += self.hist[i][j] * weight
                    result[move][1] += 1

        for choice in result:
            if result[choice][1] > 0:
                result[choice][0] /= float(result[choice][1])

        final_decision = weighted_choice(result)

        return final_decision

    def is_opposite_to_last(self, move):
        return self.opposite_move[self.last_move] == move

    def get_transition_functions(self, direction):
        v = 0 + (1 if direction == Action.DOWN else 0) - (1 if direction == Action.UP else 0)
        h = (1 if direction == Action.RIGHT else 0) - (1 if direction == Action.LEFT else 0)

        return [
            [lambda i, j: [(i + v) % self.height, (j + h) % self.width], self.p],
            [lambda i, j: [(i + v - 1) % self.height, (j + h) % self.width], self.p_d],
            [lambda i, j: [(i + v + 1) % self.height, (j + h) % self.width], self.p_d],
            [lambda i, j: [(i + v) % self.height, (j + h + 1) % self.width], self.p_d],
            [lambda i, j: [(i + v) % self.height, (j + h - 1) % self.width], self.p_d],
        ]

    def move_histo(self, direction):
        old = cp.deepcopy(self.hist)

        for y in range(self.height):
            for x in range(self.width):
                self.hist[y][x] = 0

        transitions = self.get_transition_functions(direction)

        for i in range(self.height):
            for j in range(self.width):
                # print '{0},{1}'.format(i,j)
                for trans in transitions:
                    coords = trans[0](i, j)
                    self.hist[coords[0]][coords[1]] += old[i][j] * float(trans[1])

    def apply_sense_on_histo(self, sense):
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] == 'W':
                    self.hist[i][j] = 0
                else:
                    self.hist[i][j] *= self.conditional_probs[sense, self.map[i][j]]

    def normalize_hist(self):
        max_val = max([max(line) for line in self.hist])

        for i in range(self.height):
            for j in range(self.width):
                self.hist[i][j] = max(self.hist[i][j] / max_val, 0.00000001)

    def find_exit_coord(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.map[i][j] == 'W':
                    self.exit_coords = [i, j]
                    return
