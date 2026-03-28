from cycler import K
from matplotlib.pylab import randint
from numpy import ones, zeros, random
from openai import NoneType
from openenv.core.env_server.types import State
from sklearn.metrics import mean_squared_error as MSE

random.seed(123)


OBJECTS = {
    "book": {"dims": [4, 4, 2], "stack": True},
    "penstand": {"dims": [2, 2, 4], "stack": True},
    "bottle": {"dims": [2, 2, 6], "stack": False},
    "pen": {"dims": [1, 1, 4], "stack": False},
    "pencil": {"dims": [1, 1, 6], "stack": False},
    "eraser": {"dims": [2, 1, 1], "stack": False},
    "powerbank": {"dims": [4, 2, 1], "stack": False},
    "mobile": {"dims": [4, 2, 1], "stack": False},
    "laptop": {"dims": [6, 4, 1], "stack": True},
    "monitor": {"dims": [6, 4, 2], "stack": False},
    "keyboard": {"dims": [6, 2, 1], "stack": False},
    "mouse": {"dims": [4, 2, 1], "stack": False},
    "headphones": {"dims": [4, 4, 2], "stack": False},
    "charger": {"dims": [2, 2, 1], "stack": False},
    "notebook": {"dims": [4, 4, 1], "stack": True},
    "folder": {"dims": [4, 4, 1], "stack": True},
    "backpack": {"dims": [6, 4, 2], "stack": False},
    "pouch": {"dims": [4, 4, 2], "stack": False},
}

OBJECT_NAMES = [
    "book",
    "penstand",
    "bottle",
    "pen",
    "pencil",
    "eraser",
    "powerbank",
    "mobile",
    "laptop",
    "monitor",
    "keyboard",
    "mouse",
    "headphones",
    "charger",
    "notebook",
    "folder",
    "backpack",
    "pouch",
]


def initDimentions(obj):
    dims = obj.get("dims")
    if dims is None:
        return []

    return ones(dims, dtype=int).tolist()


def initGrid():
    sizeX, sizeY, sizeZ = randint(8, 12), randint(8, 12), randint(8, 12)
    grid = zeros((sizeX, sizeY, sizeZ), dtype=int).tolist()

    numObjs = randint(3, len(OBJECT_NAMES) + 1)
    chosenNames = random.choice(OBJECT_NAMES, size=numObjs, replace=False)

    placed = {}

    for name in chosenNames:
        obj = OBJECTS.get(name)

        dimX, dimY, dimZ = obj["dims"]

        if dimX > sizeX or dimY > sizeY or dimZ > sizeZ:
            continue

        isPlaced = False
        tryPlaced = 0

        while not isPlaced and tryPlaced < 100:
            posX = randint(0, sizeX - dimX + 1)
            posY = randint(0, sizeY - dimY + 1)
            posZ = 0

            canPlace = True
            for i in range(dimX):
                for j in range(dimY):
                    for k in range(dimZ):
                        if (
                            grid[posX + i][posY + j][posZ + k] != 0
                            and obj["stack"] == False
                        ):
                            canPlace = False
                            break
                        else:
                            canPlace = True
                    if not canPlace:
                        break
                if not canPlace:
                    break

            if canPlace:
                for i in range(dimX):
                    for j in range(dimY):
                        for k in range(dimZ):
                            if (
                                obj["stack"]
                                and grid[posX + i][posY + j][posZ + k] > 0
                                and posZ + k + 1 < sizeZ
                            ):
                                grid[posX + i][posY + j][posZ + k + 1] += 1
                            else:
                                grid[posX + i][posY + j][posZ + k] += 1

            placed[name] = (posX, posY, posZ, obj["stack"])
            isPlaced = True

    return (grid, placed)


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
    for obj_name, pos in objects.items():
        obj = OBJECTS.get(obj_name)
        if obj is None:
            reward -= reward_per_obj_placed
            continue

        objGrid = initDimentions(obj)
        placement_failed = False

        for i in range(len(objGrid)):
            for j in range(len(objGrid[0])):
                for k in range(len(objGrid[0][0])):
                    if (
                        pos[0] + i >= len(dims)
                        or pos[1] + j >= len(dims[0])
                        or pos[2] + k >= len(dims[0][0])
                    ):
                        reward -= reward_per_obj_placed
                        placement_failed = True
                        break

                    if dims[pos[0] + i][pos[1] + j][pos[2] + k] > 0 and pos[3] == False:
                        reward -= reward_per_obj_placed
                        placement_failed = True
                        break

                    elif (
                        dims[pos[0] + i][pos[1] + j][pos[2] + k] > 0 and pos[3] == True
                    ):
                        if pos[2] + k + 1 <= len(objGrid[0][0]):
                            dims[pos[0] + i][pos[1] + j][pos[2] + k + 1] += 1
                            reward += (
                                weight[pos[0] + i][pos[1] + j][pos[2] + k + 1]
                                * reward_per_obj_placed
                            )
                        else:
                            reward -= reward_per_obj_placed
                            placement_failed = True

                        break

                    else:
                        dims[pos[0] + i][pos[1] + j][pos[2] + k] = 1
                        reward += (
                            reward_per_obj_placed
                            * weight[pos[0] + i][pos[1] + j][pos[2] + k]
                        )
                if placement_failed:
                    break
            if placement_failed:
                break

        if not placement_failed:
            state.ObjectsPresent[obj_name] = pos

    return reward


def findobject(objects, state):
    reward = 0.0
    objs = []
    for obj_found, pos_found in objects.items():
        pos_real = state.ObjectsPresent.get(obj_found)
        if pos_real is None:
            reward -= 10.0
            continue

        if pos_found == pos_real:
            reward += 10.0
            objs.append(obj_found)
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
