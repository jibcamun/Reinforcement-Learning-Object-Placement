from matplotlib.pylab import randint
from numpy import ones, zeros, random, shape
from openenv.core.env_server.types import State
from sklearn.metrics import mean_squared_error as MSE

random.seed(123)


def initDimentions(obj):
    dims = obj.get("dims")
    if dims is None:
        return []

    return ones(dims, dtype=int).tolist()


def initGrid():
    grid = zeros((randint(5, 11), randint(5, 11), randint(5, 11)), dtype=int).tolist()
    return grid


def initWeightedGrid():
    grid = random.uniform(0, 1, (randint(5, 11), randint(5, 11), randint(5, 11)))

    x_mid = grid.shape[0] // 2
    x_span = grid.shape[0] // 4
    y_front = grid.shape[1] // 3

    grid[x_mid - x_span : x_mid + x_span, :y_front, :] *= 0.2

    return grid


def place(objects, state):
    dims = state.currentGrid
    weight = state.weightedGrid
    reward = 0.0
    totalObjs = len(objects)
    reward_per_obj_placed = 90.0 / totalObjs
    for obj, pos in objects.items():
        objGrid = initDimentions(obj)

        for i in range(len(objGrid)):
            for j in range(len(objGrid[0])):
                for k in range(len(objGrid[0][0])):
                    if (
                        pos[0] + i > len(dims) - 1
                        and pos[1] + j > len(dims[0]) - 1
                        and pos[2] + k > len(dims[0][0]) - 1
                    ):
                        reward -= reward_per_obj_placed

                    if dims[pos[0] + i][pos[1] + k][pos[2] + j] > 0 and pos[3] == False:
                        reward -= reward_per_obj_placed
                        break
                    elif (
                        dims[pos[0] + i][pos[1] + k][pos[2] + j] > 0 and pos[3] == True
                    ):
                        dims[pos[0] + i][pos[1] + k][pos[2] + j] += 1
                        reward += (
                            weight[pos[0] + i][pos[1] + k][pos[2] + j]
                            * reward_per_obj_placed
                        )
                        break
                    else:
                        dims[pos[0] + i][pos[1] + k][pos[2] + j] = 1
                        reward += (
                            reward_per_obj_placed
                            * weight[pos[0] + i][pos[1] + k][pos[2] + j]
                        )

    return reward


def findobject(objects, state):
    reward = 0.0
    objs = []
    for (obj_found, pos_found), (obj_real, pos_real) in zip(
        objects.items(), state.ObjectsPresent.items()
    ):
        if pos_found == pos_real:
            if obj_found == obj_real:
                reward += 10.0
                objs.append(obj_found)
            else:
                reward -= 10.0
        else:
            rmse = MSE(pos_real[:3], pos_found[:3])
            reward -= rmse

        if pos_found[3] != pos_real[3]:
            reward -= 5.0
        else:
            reward += 5.0

    for obj in objs:
        state.objectsLeft.remove(obj)
        state.objectsFound.append(obj)

    return reward
