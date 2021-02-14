import numpy as np


def local_maximum_3d(candidate_coords,array,local_box_size):

    local_box_size = 14
    num_candidate = len(candidate_coords)

    candidate_x_v2 = []
    candidate_y_v2 = []
    candidate_z_v2 = []


    for nn,(nx,ny,nz) in enumerate(candidate_coords):
        starting_value = array[nx,ny,nz]
        tmp_x = nx
        tmp_y = ny
        tmp_z = nz
        for iz in range(local_box_size):
            for iy in range(local_box_size):
                for ix in range(local_box_size):

                    try:
                        compare_value = array[nx-int(local_box_size/2)+ix+1,ny-int(local_box_size/2)+iy+1,nz-int(local_box_size/2)+iz+1]

                        if starting_value < compare_value:
                            tmp_x = nx-int(local_box_size/2)+ix+1
                            tmp_y = ny-int(local_box_size/2)+iy+1
                            tmp_z = nz-int(local_box_size/2)+iz+1
                            starting_value = compare_value 
                        else:
                            pass
                    except:
                        pass

        candidate_x_v2.append(tmp_x)
        candidate_y_v2.append(tmp_y)
        candidate_z_v2.append(tmp_z)
 

    coords = set(list(zip(candidate_x_v2,candidate_y_v2,candidate_z_v2)))
    coords = np.array(sorted(coords, key = lambda x:x[2]))

        
    return coords

