from matplotlib.pylab import randint
from numpy import ones, zeros, random
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


def appendRewardFeedback(state, feedback, reward):
    state.rewardFeedback.append(feedback)
    state.rewardList.append(reward)


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


def initWeightedGrid(shape=None):
    if shape is None:
        shape = (randint(8, 12), randint(8, 12), randint(8, 12))

    grid = random.uniform(0, 1, shape)

    x_mid = grid.shape[0] // 2
    x_span = grid.shape[0] // 4
    y_front = grid.shape[1] // 3

    grid[x_mid - x_span : x_mid + x_span, :y_front, :] *= 0.2

    return grid


def _get_weight_value(weight, x, y, z):
    if not weight or not weight[0] or not weight[0][0]:
        return 0.0

    if (
        x < 0
        or y < 0
        or z < 0
        or x >= len(weight)
        or y >= len(weight[0])
        or z >= len(weight[0][0])
    ):
        return 0.0

    return weight[x][y][z]


def place(segment, objects, state):
    dims = state.currentGrid
    weight = state.weightedGrid
    objsPresent = state.ObjectsPresent

    reward = 0.0
    totalObjs = len(objects)
    reward_per_obj_placed = 45.0 / totalObjs

    if segment:
        appendRewardFeedback(
            state, "Placing objects with segmentation is not allowed.", -60.0
        )
        return -60.0

    for obj_name, pos in objects.items():

        obj = OBJECTS.get(obj_name)
        if obj is None:
            appendRewardFeedback(
                state, f"Object '{obj_name}' is not recognized.", -reward_per_obj_placed
            )
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
                        appendRewardFeedback(
                            state,
                            f"Object '{obj_name}' placement is out of bounds.",
                            -reward_per_obj_placed,
                        )
                        placement_failed = True
                        break

                    if dims[pos[0] + i][pos[1] + j][pos[2] + k] > 0 and pos[3] == False:
                        reward -= reward_per_obj_placed
                        appendRewardFeedback(
                            state,
                            f"Object '{obj_name}' placement overlaps with another object and stacking is not allowed.",
                            -reward_per_obj_placed,
                        )
                        placement_failed = True
                        break

                    elif (
                        dims[pos[0] + i][pos[1] + j][pos[2] + k] > 0 and pos[3] == True
                    ):
                        if pos[2] + k + 1 < len(dims[0][0]):
                            dims[pos[0] + i][pos[1] + j][pos[2] + k + 1] += 1
                            bonus = (
                                _get_weight_value(
                                    weight,
                                    pos[0] + i,
                                    pos[1] + j,
                                    pos[2] + k + 1,
                                )
                                * reward_per_obj_placed
                            )
                            reward += bonus
                            appendRewardFeedback(
                                state,
                                f"Object '{obj_name}' placed with stacking. Bonus: {bonus:.2f}",
                                bonus,
                            )
                        else:
                            reward -= reward_per_obj_placed
                            appendRewardFeedback(
                                state,
                                f"Object '{obj_name}' placement failed. No space for stacking.",
                                -reward_per_obj_placed,
                            )
                            placement_failed = True

                        break

                    else:
                        dims[pos[0] + i][pos[1] + j][pos[2] + k] = 1
                        bonus = reward_per_obj_placed * _get_weight_value(
                            weight, pos[0] + i, pos[1] + j, pos[2] + k
                        )
                        reward += bonus
                        appendRewardFeedback(
                            state,
                            f"Object '{obj_name}' placed successfully. Bonus: {bonus:.2f}",
                            bonus,
                        )
                if placement_failed:
                    break
            if placement_failed:
                break

        if not placement_failed:
            state.ObjectsPlaced[obj_name] = pos
            state.numberPlaced += 1
            try:
                if objsPresent[obj_name] == state.ObjectsPlaced[obj_name]:
                    reward -= 45.0 / totalObjs
                    appendRewardFeedback(
                        state,
                        f"Object '{obj_name}' is being placed in the same location",
                        -reward_per_obj_placed,
                    )
            except KeyError:
                reward -= reward_per_obj_placed
                appendRewardFeedback(
                    state,
                    f"Object '{obj_name}' is present in the environment, but is placed in same location as originally found.",
                    -reward_per_obj_placed,
                )

                continue

    return reward


def findobject(segment, objects, state):

    if not segment or segment is None:
        appendRewardFeedback(
            state, "Finding objects without segmentation is not allowed.", -60.0
        )
        return -60.0

    if state.ObjectsPresent == state.objectsFound:
        appendRewardFeedback(
            state,
            "No point in finding more objects as all are already found Make the IsSegement attribute false and execute the place method.",
            -60.0,
        )
        return -60.0

    reward = 0.0
    glMetric = 45.0 / len(state.ObjectsPresent)
    objs = []
    for obj_found, pos_found in objects.items():
        pos_real = state.ObjectsPresent.get(obj_found)
        if pos_real is None:
            reward -= glMetric
            appendRewardFeedback(
                state, f"Object '{obj_found}' not found in the environment.", -glMetric
            )
            continue

        if pos_found == pos_real:
            reward += glMetric
            appendRewardFeedback(
                state,
                f"Object '{obj_found}' found with correct position and stacking.",
                glMetric,
            )
            objs.append(obj_found)
        else:
            mse = MSE(pos_real[:3], pos_found[:3])
            reward -= mse
            appendRewardFeedback(
                state,
                f"Object '{obj_found}' found with incorrect position. MSE: {mse:.2f}",
                -mse,
            )

        if pos_found[3] != pos_real[3]:
            reward -= glMetric / 4.0
            appendRewardFeedback(
                state,
                f"Object '{obj_found}' found with incorrect stacking. Penalty: {glMetric / 4.0}",
                -glMetric / 4.0,
            )
        else:
            reward += glMetric / 4.0
            appendRewardFeedback(
                state,
                f"Object '{obj_found}' found with correct stacking. Bonus: {glMetric / 4.0}",
                glMetric / 4.0,
            )

    for obj in objs:
        state.objectsLeft.remove(obj)
        state.objectsFound.append(obj)

    return reward


def _remove_object(state, obj_name):
    reward = 0
    try:
        pos = state.ObjectsPlaced.pop(obj_name)
    except KeyError:
        reward -= 45.0 / len(state.ObjectsPresent)
        appendRewardFeedback(
            state,
            f"Object '{obj_name}' is not placed in the environment.",
            -reward,
        )
        return reward

    state.numberPlaced -= 1
    dims = state.currentGrid
    obj = OBJECTS.get(obj_name)
    objGrid = initDimentions(obj)

    for i in range(len(objGrid)):
        for j in range(len(objGrid[0])):
            for k in range(len(objGrid[0][0])):
                if dims[pos[0] + i][pos[1] + j][pos[2] + k] > 0:
                    dims[pos[0] + i][pos[1] + j][pos[2] + k] -= 1


def adjustment(segment, action, state):
    objsPlaced = state.ObjectsPlaced

    if segment:
        appendRewardFeedback(
            state, "Placing objects with segmentation is not allowed.", -60.0
        )
        return -60.0

    try:
        initPos = objsPlaced[action[0]]
        name = action[0]
    except KeyError:
        reward_per_obj_placed = 45.0 / len(state.ObjectsPresent)
        appendRewardFeedback(
            state,
            f"Object '{action[0]}' is not placed in the environment, so it cannot be adjusted.",
            -reward_per_obj_placed,
        )
        return -reward_per_obj_placed

    if action[1] == "RIGHT":
        _remove_object(state, name)
        newPos = (initPos[0] + 1, initPos[1], initPos[2], initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved right successfully.",
            reward,
        )
        return reward
    elif action[1] == "LEFT":
        _remove_object(state, name)
        newPos = (initPos[0] - 1, initPos[1], initPos[2], initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved left successfully.",
            reward,
        )
        return reward
    elif action[1] == "UP":
        _remove_object(state, name)
        newPos = (initPos[0], initPos[1] + 1, initPos[2], initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved up successfully.",
            reward,
        )
        return reward
    elif action[1] == "DOWN":
        _remove_object(state, name)
        newPos = (initPos[0], initPos[1] - 1, initPos[2], initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved down successfully.",
            reward,
        )
        return reward
    elif action[1] == "FORWARD":
        _remove_object(state, name)
        newPos = (initPos[0], initPos[1], initPos[2] + 1, initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved forward successfully.",
            reward,
        )
        return reward
    elif action[1] == "BACKWARD":
        _remove_object(state, name)
        newPos = (initPos[0], initPos[1], initPos[2] - 1, initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' moved backward successfully.",
            reward,
        )
        return reward
    elif action[1] == "ROTATE":
        _remove_object(state, name)
        newPos = (initPos[0], initPos[2], initPos[1], initPos[3])
        reward = place(segment, {name: newPos}, state)
        appendRewardFeedback(
            state,
            f"Object '{name}' rotated successfully.",
            reward,
        )
        return reward
    else:
        reward_per_obj_placed = 45.0 / len(state.ObjectsPresent)
        appendRewardFeedback(
            state,
            f"Invalid adjustment direction '{action[1]}'. Valid directions are RIGHT, LEFT, UP, DOWN, FORWARD, BACKWARD, ROTATE.",
            -reward_per_obj_placed,
        )
        return -reward_per_obj_placed
