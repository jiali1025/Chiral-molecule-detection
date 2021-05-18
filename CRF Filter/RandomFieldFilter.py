import numpy
import torch


class RF_filter:
    """This filter will be applied on the top of faster R-CNN output. It is expected to learn the molecular pattern on
    the densely packed STM image, and to increase the accuracy less certain molecules by the chirality of
    neighboring molecules based on a conditional random field constructed from learned molecular pattern.
    """

    def __init__(self, instances):

        self.distance_lb = 0.7
        self.distance_ub = 1.5  # neighbor detection threshold

        self.instances = instances  # output from faster R-CNN model
        self.num = None  # num of molecules found by faster R-CNN
        self.mesh = self.reformat()  # construct a mesh based on centers of molecules
        self.remove_odd_size()  # remove prediction which is not a molecule
        self.distance = self.inter_distance()  # found inter-molecular distance of the system
        self.prob = self.retrieve_probability()  # construct a conditional random field

    def reformat(self):
        """construct a mesh based on centers of molecules
        each mesh[index] contains a dict for one molecule {location, class, probability} which is given by faster R-CNN
        """

        pred_boxes = self.instances['pred_boxes'].tensor
        pred_class = self.instances['pred_classes']
        pred_scores = self.instances['scores']

        data = []
        self.num = len(pred_boxes)
        for i in range(self.num):
            loc = numpy.array([pred_boxes[i][0] + pred_boxes[i][2], pred_boxes[i][1] + pred_boxes[i][3]]) / 2.0

            data_dict = {}
            data_dict["loc"] = loc
            data_dict["class"] = pred_class[i]
            data_dict["score"] = pred_scores[i]

            data.append(data_dict)
        return data

    def remove_odd_size(self):
        """remove prediction which is not a molecule based on the size of bbox"""

        n_sample = min(self.num, 30)
        d_sample = numpy.zeros(n_sample)
        index_sample = numpy.random.randint(0, self.num, n_sample)

        for i in range(n_sample):
            bbox = self.instances['pred_boxes'].tensor[index_sample[i]]
            d_sample[i] = (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2

        diamter = numpy.median(d_sample)

        for i in reversed(range(self.num)):
            bbox = self.instances['pred_boxes'].tensor[i]
            d_i = (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
            d_i_reduce = d_i / diamter

            if d_i_reduce < 0.8 or d_i_reduce > 1.2:
                del self.mesh[i]
                boxes = self.instances['pred_boxes'].tensor
                self.instances['pred_boxes'].tensor = torch.cat((boxes[:i], boxes[i + 1:]))
                classes = self.instances['pred_classes']
                self.instances['pred_classes'] = torch.cat((classes[:i], classes[i + 1:]))
                scores = self.instances['scores']
                self.instances['scores'] = torch.cat((scores[:i], scores[i + 1:]))

        self.num = len(self.mesh)

    def calc_distance(self, index1, index2):
        """ calculate the distance between two molecules with index 1 and 2
        """

        d2 = (self.mesh[index1]['loc'][0] - self.mesh[index2]['loc'][0]) ** 2 + (
                self.mesh[index1]['loc'][1] - self.mesh[index2]['loc'][1]) ** 2
        d = d2 ** 0.5

        return d

    def inter_distance(self):
        """ calculate the inter-molecular distance in the densely packed system by sampling 30 molecules. The
        inter-molecular distance is determined by the average of the 20 minimum distances between each sampled
        molecule and all other molecules
        """

        sample_size = min(30, self.num)
        sample = numpy.random.randint(0, self.num, sample_size)

        r_min = numpy.empty(sample_size)

        for i in sample:
            distance = numpy.empty(self.num - 1)
            count = 0
            for j in range(self.num):
                if i == j:
                    continue
                d = self.calc_distance(i, j)
                distance[count] = d
                count = count + 1

            r_min = min(distance)

        r = numpy.median(r_min)

        return r

    def find_neighbor(self, mol_index):
        """Find all the neighboring molecules of the molecule with a certain index based on the inter-molecular
        distance """

        neighbor_index = []

        for i in range(self.num):
            d = self.calc_distance(mol_index, i)
            d_reduce = d / self.distance

            if self.distance_lb < d_reduce < self.distance_ub:
                neighbor_index.append(i)

        return neighbor_index

    def get_neighbor_info(self, neighbor_index):
        """Retrieve the neighbor info of molecule with the index"""

        C0_num = 0  # number of neighbors with chirality C0
        C1_num = 0  # number of neighbors with chirality C1
        neighbor_num = len(neighbor_index)

        for i in range(neighbor_num):
            if self.mesh[neighbor_index[i]]["class"] == 0:
                C0_num = C0_num + 1
            elif self.mesh[neighbor_index[i]]["class"] == 1:
                C1_num = C1_num + 1

        return neighbor_num, C0_num, C1_num

    def retrieve_probability(self):
        """Construct a conditional random field. Scan through all molecules and determine the probability for a
        molecule to be each chirality while its neighbors are observed to be in a certain state"""

        """Based on the domain knowledge, the molecule has 6 neighbors. prob: [2,6,6] will be used for record the 
        probability for a molecule to be R or L while observing different neighboring conditions . P(x=R|neighbor R and L) 
        and P(x=L|neighbor R and L) """

        prob_relax = 0.05  # relax the probability to allow abnormal molecular patterns

        occur = numpy.zeros([2, 7, 7])
        prob = numpy.zeros([2, 7, 7])

        for i in range(self.num):
            if self.mesh[i]["score"] < 0.9:
                continue  # only trust info from high-confidence molecules

            neighbor_i = self.find_neighbor(i)

            neighbor_num, neighbor_C0, neighbor_C1 = self.get_neighbor_info(neighbor_i)

            if neighbor_num > 6:
                continue

            mol_class = self.mesh[i]["class"]
            occur[mol_class, neighbor_C0, neighbor_C1] += 1

        occur[occur <= 3] = 0

        prob = numpy.divide(occur, numpy.sum(occur, 0))
        prob = numpy.clip(prob, prob_relax, 1 - prob_relax)

        return prob

    def update_score(self, index):
        """ update the probability of low-confidence predictions by using info from neighboring molecules"""

        neighbor_index = self.find_neighbor(index)
        neighbor_num, C0_num, C1_num = self.get_neighbor_info(neighbor_index)

        if neighbor_num > 6:
            return 0

        mol_class = self.mesh[index]["class"]
        prob_filter = self.prob[mol_class, C0_num, C1_num]

        if not numpy.isnan(prob_filter):
            prob_model = self.mesh[index]["score"]

            Z = prob_model * prob_filter + (1 - prob_model) * (1 - prob_filter)
            prob_update = prob_model * prob_filter / Z

            if prob_update > 0.5:
                self.mesh[index]["score"] = prob_update
            else:
                self.mesh[index]["class"] = 1 - mol_class
                self.mesh[index]["score"] = 1 - prob_update

    def instance_update(self):
        """ update the filtered result"""

        new_instances = self.instances

        for i in range(self.num):
            new_instances["scores"][i] = self.mesh[i]["score"]
            new_instances["pred_classes"][i] = self.mesh[i]["class"]

        return new_instances

    def filter_main(self, prob_thresh=0.7):
        """ Use info from neighboring molecules to ascertain molecules with low probability level predicted by faster
        R-CNN """

        for i in range(self.num):
            score_i = self.mesh[i]["score"]
            if score_i > prob_thresh:
                continue

            self.update_score(i)

        filter_results = self.instance_update()

        return filter_results
