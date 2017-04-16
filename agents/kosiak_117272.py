from action import Action
import numpy as np
import copy as cp
import operator


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
        self.map = [[str(x) for x in line ] for line in areaMap]

        self.find_exit_coord()

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

        self.bayes_probs = {}
        bayes_sum = 0

        for r in [False, True]:
            for f in ['.', 'J']:
                self.bayes_probs[(f, r)] = self.conditional_probs[r, f] * self.field_probs[f] / self.result_probs[r]
                bayes_sum += self.bayes_probs[(f, r)]

        for r in [False, True]:
            for f in ['.', 'J']:
                self.bayes_probs[(f, r)] /= bayes_sum

        self.hist = []
        for y in range(self.height):
            self.hist.append([])
            for x in range(self.width):
                self.hist[y].append(1.)

        self.hist[self.exit_coords[0]][self.exit_coords[1]] = 0

        # output

        print 'params p={0} pj={1} pn={2}'.format(self.p, self.pj, self.pn)

        print 'Map\n{0}'.format('\n'.join(['     '.join(line) for line in self.map]))
        self.print_pretty_hist()

        print 'conditional_probs'
        for entry in self.conditional_probs:
            print '{0} : {1}'.format(entry, self.conditional_probs[entry])

        print 'result probs : {0}'.format(self.result_probs)

        print 'Bayess probs'
        for entry in self.bayes_probs:
            print '{0} : {1}'.format(entry, self.bayes_probs[entry])

        print '================start===================='

        return

    def sense(self, sensor):
        print '=================================MOVE START=============================='

        print "Sense {0} = {1}".format(self.times_moved, sensor)
        self.apply_sense_on_histo(sensor)
        self.print_pretty_hist()

    # nie zmieniac naglowka metody, tutaj agent decyduje w ktora strone sie ruszyc,
    # funkcja MUSI zwrocic jedna z wartosci [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    def move(self):
        self.times_moved += 1
        current_decision = self.calculate_direction()
        print 'Moving = {0}'.format(current_decision)

        self.move_histo(current_decision)
        self.print_pretty_hist()

        print 'Normalizing'
        self.normalize_hist()

        self.print_pretty_hist()
        print '=================================MOVE {0} END=============================='.format(self.times_moved)

        # exit()
        return current_decision

    # nie zmieniac naglowka metody, tutaj agent udostepnia swoj histogram (ten z filtru
    # histogramowego), musi to byc tablica (lista list, krotka krotek...) o wymarach takich jak
    # plansza, pobranie wartosci agent.histogram()[y][x] zwraca prawdopodobienstwo stania na polu
    # w wierszu y i kolumnie x
    def histogram(self):
        return self.hist

    # =============================================================================================
    # =======================================CUSTOM================================================
    # =============================================================================================

    def calculate_direction(self):
        result = {act: 0 for act in [Action.UP, Action.DOWN, Action.RIGHT, Action.LEFT]}

        for i in range(self.width):
            for j in range(self.height):
                if [i, j] == self.exit_coords:
                    continue

                avail_decisions = []

                v_d_inner = self.exit_coords[1] - j

                # print 'v_d_inner={0}'.format(v_d_inner)

                if v_d_inner < 0:
                    avail_decisions.append([abs(v_d_inner), Action.LEFT])
                    avail_decisions.append([self.width - i + self.exit_coords[1] , Action.RIGHT])
                elif v_d_inner > 0:
                    avail_decisions.append([v_d_inner, Action.RIGHT])
                    avail_decisions.append([i + self.height - self.exit_coords[1] , Action.LEFT])

                h_d_inner = self.exit_coords[0] - i

                if h_d_inner < 0:
                    avail_decisions.append([abs(h_d_inner), Action.UP])
                    avail_decisions.append([self.width - j + self.exit_coords[0], Action.DOWN])
                elif h_d_inner > 0:
                    avail_decisions.append([h_d_inner, Action.DOWN])
                    avail_decisions.append([i + self.width - self.exit_coords[0], Action.UP])

                min_d = min(avail_decisions, key=lambda x:x[0])[0]

                # print 'coord [{0},{1}]'.format(i,j)
                # print min_d
                # print 'all {0}'.format(avail_decisions)

                avail_decisions = [ad for ad in avail_decisions if ad[0] == min_d]
                # print 'min {0}'.format(avail_decisions)

                for av in avail_decisions:
                    result[av[1]] += self.hist[i][j]

        return max(result.iteritems(), key=operator.itemgetter(1))[0]

    def get_transition_functions(self, direction):
        v = 0 + (1 if direction == Action.UP else 0) - (1 if direction == Action.DOWN else 0)
        h = (1 if direction == Action.RIGHT else 0) - (1 if direction == Action.LEFT else 0)

        print 'v={0} h={1}'.format(v,h)

        return [
            [lambda i, j: [(i + v) % self.width, (j + h) % self.height], self.p],
            [lambda i, j: [(i + v - 1) % self.width, (j + h) % self.height], self.p_d],
            [lambda i, j: [(i + v + 1) % self.width, (j + h) % self.height], self.p_d],
            [lambda i, j: [(i + v) % self.width, (j + h + 1) % self.height], self.p_d],
            [lambda i, j: [(i + v) % self.width, (j + h - 1) % self.height], self.p_d],
        ]

    def move_histo(self, direction):
        old = cp.deepcopy(self.hist)

        print 'old[4,9]={0}'.format(old[4][9])
        print 'old[4,6]={0}'.format(old[4][6])

        for y in range(self.height):
            for x in range(self.width):
                self.hist[y][x] = 0

        transitions = self.get_transition_functions(direction)

        for i in range(self.width):
            for j in range(self.height):
                # print '{0},{1}'.format(i,j)
                for trans in transitions:
                    coords = trans[0](i, j)
                    # print '{0},{1} p={2} r={3}'.format(coords[0], coords[1], trans[1], old[i][j] )
                    self.hist[coords[0]][coords[1]] += old[i][j] * float(trans[1])

    def apply_sense_on_histo(self, sense):
        for i in range(self.width):
            for j in range(self.height):
                if self.map[i][j] == 'W':
                    continue

                self.hist[i][j] *= self.bayes_probs[self.map[i][j], sense]

    def normalize_hist(self):
        max_val = max([max(line) for line in self.hist])
        # print 'max value {0}'.format(max_val)

        for i in range(self.width):
            for j in range(self.height):
                self.hist[i][j] = max(self.hist[i][j]/ max_val, 0.001)

    def print_pretty_hist(self):
        # return
        print "\n".join([" ".join(["{0:.3f}".format(el) for el in line]) for line in self.hist])

    def find_exit_coord(self):
        for i in range(self.width):
            for j in range(self.height):
                if self.map[i][j] == 'W':
                    self.exit_coords = [i,j]
                    return
