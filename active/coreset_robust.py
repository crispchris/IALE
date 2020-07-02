import time

import gurobipy as gb
import np as np
from torch.utils.data import ConcatDataset

from active.strategy import Strategy


def solve_fac_loc(xx, yy, subset, number_images, budget):

    model = gb.Model("k-center")
    x = {}
    y = {}
    z = {}
    for image_idx in range(number_images):
        # z_i: is a loss
        z[image_idx] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(image_idx))

    m = len(xx)
    for node_idx in range(m):
        _x = xx[node_idx]
        _y = yy[node_idx]
        # y_i = 1 means i is facility, 0 means it is not
        if _y not in y:
            if _y in subset:
                y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
            else:
                y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
        # if not _x == _y:
        x[_x, _y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x, _y))
    model.update()

    coef = [1 for j in range(number_images)]
    var = [y[j] for j in range(number_images)]
    model.addConstr(gb.LinExpr(coef, var), "=", rhs=budget + len(subset), name="k_center")

    yyy = {}
    for node_idx in range(m):
        _x = xx[node_idx]
        _y = yy[node_idx]

        if _x not in yyy:
            yyy[_x] = []
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

        # if not _x == _y:
        model.addConstr(x[_x, _y], "<", y[_y], name="Strong_{},{}".format(_x, _y))

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            # if not _x==_y:
            coef.append(1)
            var.append(x[_x, _y])
        coef.append(1)
        var.append(z[_x])
        model.addConstr(gb.LinExpr(coef, var), "=", 1, name="Assign{}".format(_x))
    model.__data = x, y, z
    return model


class RobustCoresetSampling(Strategy):
    name = 'robustcoreset'

    def __init__(self, dataset_pool, valid_dataset, test_dataset, device='cuda:0'):
        super(RobustCoresetSampling, self).__init__(dataset_pool, [], valid_dataset, test_dataset)
        self.min_distances = None

    def query(self, n, model, train_dataset, pool_dataset, budget=10000):
        device = model.state_dict()['softmax.bias'].device

        full_dataset = ConcatDataset([pool_dataset, train_dataset])
        pool_len = len(pool_dataset)

        self.embeddings = self.get_embeddings(model, device, full_dataset)

        # Calc distance matrix
        num_images = self.embeddings.shape[0]
        dist_mat = self.calc_distance_matrix(num_images)

        # We need to get k centers start with greedy solution
        upper_bound = gb.UB
        lower_bound = upper_bound / 2.0
        max_dist = upper_bound

        _x, _y = np.where(dist_mat <= max_dist)
        _distances = dist_mat[_x, _y]
        subset = [i for i in range(1)]
        model = solve_fac_loc(_x, _y, subset, num_images, budget)
        # model.setParam( 'OutputFlag', False )
        x, y, z = model.__data
        delta = 1e-7

        while upper_bound - lower_bound > delta:
            print("State", upper_bound, lower_bound)
            current_radius = (upper_bound + lower_bound) / 2.0

            violate = np.where(_distances > current_radius)  # Point distances which violate the radius

            new_max_d = np.min(_distances[_distances >= current_radius])
            new_min_d = np.max(_distances[_distances <= current_radius])

            print("If it succeeds, new max is:", new_max_d, new_min_d)

            for v in violate[0]:
                x[_x[v], _y[v]].UB = 0  # The upper bound for points, which violate the radius are set to zero

            model.update()
            r = model.optimize()

            if model.getAttr(gb.GRB.Attr.Status) == gb.GRB.INFEASIBLE:
                failed = True
                print("Infeasible")
            elif sum([z[i].X for i in range(len(z))]) > 0:
                failed = True
                print("Failed")
            else:
                failed = False

            if failed:
                lower_bound = max(current_radius, new_max_d)
                # failed so put edges back
                for v in violate[0]:
                    x[_x[v], _y[v]].UB = 1
            else:
                print("solution found", current_radius, lower_bound, upper_bound)
                upper_bound = min(current_radius, new_min_d)
                model.write("s_{}_solution_{}.sol".format(budget, current_radius))

        idxs_labeled = np.arange(start=pool_len, stop=pool_len + len(train_dataset))

        # Perform kcenter greedy
        self.update_distances(idxs_labeled, idxs_labeled, only_new=False, reset_dist=True)
        sel_ind = []
        for _ in range(n):
            ind = np.argmax(self.min_distances)  # Get sample with highest distance
            assert ind not in idxs_labeled, "Core-set picked index already labeled"
            self.update_distances([ind], idxs_labeled, only_new=True, reset_dist=False)
            sel_ind.append(ind)

        assert len(set(sel_ind)) == len(sel_ind), "Core-set picked duplicate samples"

        remaining_ind = list(set(np.arange(pool_len)) - set(sel_ind))

        return sel_ind, remaining_ind

    def calc_distance_matrix(self, num_images):
        start = time.clock()
        dist_mat = np.matmul(self.embeddings, self.embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(num_images, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        elapsed = time.clock() - start
        print("Time spent in (distance computation) is: ", elapsed)
        return dist_mat
