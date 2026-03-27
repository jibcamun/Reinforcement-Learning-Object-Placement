from numpy import ones
from openenv.core.env_server.types import State
from app_environment import OBJECTS


def initDimentions(obj):
    dims = obj.get("dims")
    
    if dims is None:
        return []
    
    return ones(dims, dtype=int).tolist()

        
def place(objects, state):
    dims = state.currentGrid
    reward = 0.0
    totalObjs = len(objects)
    reward_per_obj_placed = 90.0/total_objs
    for obj, pos in objects.items():
        objGrid = initDimentions(obj)

        for i in range(len(objGrid)):
            for j in range(len(objGrid[0])):
                for k in range(len(objGrid[0][0])):
                    if pos[0]+i > len(dims) and pos[1] + j > len(dims[0]) and pos[2] = :
                        
                        
                    if dims[pos[0]+i][pos[1]+k][pos[2]+j] > 1 and pos[3] == False:
                        reward -= reward_per_obj_placed
                        break
                    elif dims[pos[0]+i][pos[1]+k][pos[2]+j] > 1 and pos[3] == True:
                        dims[pos[0]+i][pos[1]+k][pos[2]+j]+=1
                        reward += dims[pos[0]+i][pos[1]+k][pos[2]+j]*reward_per_obj_placed
                        break
                    else:
                        dims[pos[0]+i][pos[1]+k][pos[2]+j]=1
                        reward+=reward_per_obj_placed*dims[pos[0]+i][pos[1]+k][pos[2]+j]

    
def findobject(action, state):
    pass