'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
import distinctipy
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Defines type of noise strings used for adding noise to relative pose
NO_NOISE = 'NO_NOISE'
GAUSSIAN_ADD_NOISE = 'GAUSSIAN_ADD_NOISE'


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True
   		

# This essential puts the main function into a subroutine. 
# This allows for the investigation of noise for the relative pose by providing noise 
# parameters. It also allows for different files to be saved for noise expirements.
def ipcv_part_2_main_function(args, file_prefix, noise_added, noise_param, display_min): 
    # Stores results of running code (e.g. error estimates).
    # This is used when measuring performance as noise varies.
    col_names = ['no_spheres', 'spheres_detected', 'mean_dist_centres', 'centres_within_10_percent',
            'mean_radius_error', 'radi_within_10_percent', 'noise_factor']
    results_estimates = []

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()

    # Sets up and displays spheres in 3D visualisation.
    #####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    if display_min:
        K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

        # Rendering RGB-D frames given camera poses
        # create visualiser and get rendered views
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = K
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=img_width, height=img_height, left=0, top=0)
        for m in obj_meshes:
            vis.add_geometry(m)
        ctr = vis.get_view_control()
        for (H_wc, name, dname) in render_list:
            cam.extrinsic = H_wc
            ctr.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(name, True)
            vis.capture_depth_image(dname, True)
        vis.run()
        vis.destroy_window()
    ##################################################

    # load in the images for post processings
    img0 = cv2.imread('Images/view0.png', -1)
    dep0 = cv2.imread('Images/depth0.png', -1)
    img1 = cv2.imread('Images/view1.png', -1)
    dep1 = cv2.imread('Images/depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre and display_min:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()
    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################
    # Visualises circles given the image and the list of circles
    def visualise_circles(img, circles, name, cols=[]):
        if len(cols) == 0: # Draws green circles by default
            cols = [(0, 255, 0) for _ in range(len(circles))] 
        img_circ = np.array(img)
        for i in range(len(circles)):
            circ = circles[i]
            center, rad = circ
            img_circ = cv2.circle(img_circ, center, rad, cols[i], 2)
        # Shows circles
        if args.additional_visualisation and display_min:
            cv2.imshow('Detected Circles', img_circ)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite(file_prefix + name, img_circ)
        return img_circ

    # Detects circles in a given image
    def detect_circles(img):
        # Creates a an array of the red values of the image
        # Due to colour scheme, this proves more effective than using just grayscale.
        height, width, _ = img.shape
        img_red = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for y in range(0, height):
            for x in range(0, width):
                img_red[y, x] = 255 - img[y, x, 2]
        # Applies guassian blur to smoothen edges, reducing noise in image and improving circle detection
        img_red = cv2.GaussianBlur(img_red, (7, 7), 0)
        # Performs hough transform. Parameters selected through manual tuning.
        # The following process was roughly used: select as high of a param1 as possible that still retains enough
        # gradient information to detect all circles, with a low param2. Then increase param2 to reduce
        # false positive circles.
        # Different configurations of scene may do better and worse (e.g. large max radius for sphere etc)
        circles_img = cv2.HoughCircles(img_red, method=cv2.HOUGH_GRADIENT, dp=1,
                                    minDist=15, param1=30, param2=18,
                                        minRadius=5, maxRadius=50)
        # Returns an emtpy list if no circles detected
        if circles_img is None: return []
        circ_list = []
        for circ in circles_img[0,:]:
            center = (round(circ[0]), round(circ[1]))
            rad = round(circ[2])
            circ_list.append((center, rad))
        return circ_list
    
    circ0 = detect_circles(img0)
    circ1 = detect_circles(img1)
    img0_circles = visualise_circles(img0, circ0, 'view0circ.png')
    img1_circles = visualise_circles(img1, circ1, 'view1circ.png')
    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################
    # Visualises epipolar lines
    def visualise_epipolar_lines(img, circles, name, cols=[]):
        if len(cols) == 0:
            cols = [(255, 0, 0) for _ in range(len(circles))]
        # Draws epipolar line
        img_line = np.array(img)
        for circ_idx in range(len(circles)):
            (centerx0, centery0), _ = circles[circ_idx]
            p_0 = np.array([centerx0, centery0, f]) # Centre point in image coordinates

            u = np.matmul(fundamental_mat, p_0).reshape(3,) # Epipolar line
            # From epipolar line, generates two points so that line fits image plane.
            # Uses x_r * u_l1 + y_r * u_l2 + f * u_l3 = 0
            start_img = np.array([0, (-f * u[2]) / u[1]], dtype=np.int64)
            end_img = np.array([img_width, -(f * u[2] + u[0] * img_width) / u[1]], dtype=np.int64)
            img_line = cv2.line(img_line, start_img, end_img, cols[circ_idx], 3)

        if args.additional_visualisation and display_min:
            cv2.imshow('Epiploar Lines', img_line)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cv2.imwrite(file_prefix + name, img_line)
    # Calculates essential matrix

    # Gets camera 1 to camera 0 transformation, from which
    # we can obtain the translation and rotation
    hom_cam1_to_cam0_trans = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    rot_mat = hom_cam1_to_cam0_trans[:3, :3].T
    trans = hom_cam1_to_cam0_trans[:3, 3]

    # Adds noise to relative pose (rotation matrix and translation matrix), if specified
    if noise_added != NO_NOISE:
        if noise_added == GAUSSIAN_ADD_NOISE:
            rot_noise = np.random.normal(0, noise_param, (3, 3))
            trans_noise = np.random.normal(0, noise_param, (3))
            # Applies noise
            rot_mat += rot_noise
            trans += trans_noise

    # Gets S matrix using translation
    S = np.array([
        [0, -trans[2], trans[1]],
        [trans[2], 0, -trans[0]],
        [-trans[1], trans[0], 0]
    ])
    # E = RS
    essential_mat = np.matmul(rot_mat, S)

    # Computes M matrix
    sx = 1
    sy = 1
    M = np.array([
        [sx, 0, (- ox * sx) / f],
        [0, sy, (- oy * sy) / f],
        [0, 0, 1]
    ])
    # Computes fundamental matrix. F = M_r.T E M_l
    fundamental_mat = np.matmul(np.matmul(M.T, essential_mat), M)

    visualise_epipolar_lines(img1_circles, circ0, 'Epiploar Lines View 1.png')
    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################
    # Calculates epipolar constraint using p_hat_r.T F p_hat_l.
    # Should be zero for two perfectly matching points.
    def epi_score(circ_l, fundamental_mat, circ_r):
        (centerx_r, centery_r), _ = circ_r
        (centerx_l, centery_l), _ = circ_l
        p_l = np.array([centerx_l, centery_l, f])
        p_r = np.array([centerx_r, centery_r, f]).T
        epi = np.matmul(p_r, np.matmul(fundamental_mat, p_l))
        return epi
    
    # Uses epiploar constraint to find correspondence between spheres
    # We greedily select the correspondences. We take the circle
    # with the most votes in acc matrix from one list and compare this circle to all circles
    # in the other list. The circle that minimises the epipolar constraint is taken and then removed
    # from that list. In practice, greedy and optimal algorithm return same answer often.
    def get_correspondence_greedily(circ0, circ1):
        circ_pairs = []
        while len(circ0) > 0 and len(circ1) > 0:
            circ_l = circ0.pop(0)
            best_circ_index = 0
            best_epi_cons = epi_score(circ_l, fundamental_mat, circ1[0])
            for i in range(1, len(circ1)):
                epi_cons = epi_score(circ_l, fundamental_mat, circ1[i])
                if abs(epi_cons) <= abs(best_epi_cons):
                    best_epi_cons = epi_cons
                    best_circ_index = i
            circ_r = circ1.pop(best_circ_index)
            circ_pairs.append((circ_l, circ_r))
        return circ_pairs
    
    # Gets optimal correspondence pair by considering every possible correspondence and taking
    # set which minimises epipolar constraint. This approach is slower and uses scipy implementation
    # of hungarian algo.
    def get_correspondence_optimal(circ0, circ1):
        # Find epiploar cons for each pair of values
        epi_pairs = np.zeros((len(circ0), len(circ1)))
        for i in range(len(circ0)):
            for j in range(len(circ1)):
                epi_pairs[i, j] = abs(epi_score(circ0[i], fundamental_mat, circ1[j]))
        # Problem can be solved using hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(epi_pairs)
        circ_pairs = [(circ0[i], circ1[j]) for i, j in zip(row_indices, col_indices)]
        return circ_pairs
    
    # Gets circle correspondence. Note may wish to use greedy algorithm for when lots of circles detected
    circ_pairs = get_correspondence_optimal(circ0, circ1)
    # Visualises correspondence and also epiploar lines
    cols = distinctipy.get_colors(len(circ_pairs))
    cols = [(round(r * 255.0), round(b * 255.0), round(g * 255.0)) for (r, g, b) in cols]
    img0_circles_corr = visualise_circles(img0, [left_circ for (left_circ, _) in circ_pairs], 'view0corr.png', cols)
    img1_circles_corr = visualise_circles(img1, [right_circ for (_, right_circ) in circ_pairs], 'view1corr.png', cols)
    visualise_epipolar_lines(img1_circles_corr, [left_circ for (left_circ, _) in circ_pairs],
                              'Epiploar Lines View 1 Correspondence.png', cols)
    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################
    # Uses a p_l - b R.T p_r - T - c (p_l x R.T p_r) = 0
    # for corresponding p_l and p_r to calculate reconstructured sphere centres
    def image_coord_to_world(l, r):
        # Unpacks coords
        x_l, y_l = l
        x_r, y_r = r
        # Computes p_l using p_l = M_l p_hat_l
        p_hat_l = np.array([x_l, y_l, f])
        p_l = np.matmul(M, p_hat_l)
        # Computes - R.T p_r. Note p_r = M_r p_hat_r
        p_hat_r = np.array([x_r, y_r, f])
        p_r = np.matmul(M, p_hat_r)
        b_comp = - np.matmul(rot_mat.T, p_r)
        # Computes - p_l x R.T p_r
        c_comp = - np.cross(p_l, np.matmul(rot_mat.T, p_r))
        # Creates H matrix
        H = np.array([
            [p_l[0], b_comp[0], c_comp[0]],
            [p_l[1], b_comp[1], c_comp[1]],
            [p_l[2], b_comp[2], c_comp[2]]
        ])
        # Computes a, b and c using [a, b, c].T = H^-1 T
        [a, b, _] = np.matmul(np.linalg.inv(H), trans)
        # Reconstructed point p_recon = (a p_l + b R.T p_r + T) / 2
        p_recon = ((a * p_l) + (b * np.matmul(rot_mat.T, p_r)) + trans) / 2
        # Converts reconstructed point to world coordinates from cam coordinates
        p_recon_world = transform_points(np.array([p_recon]), np.linalg.inv(H0_wc))[0]
        return p_recon_world

    recon_sphere_centres = np.zeros((len(circ_pairs), 3))
    for i, (circ_l, circ_r) in enumerate(circ_pairs):
        centre_l, _ = circ_l
        centre_r, _ = circ_r
        recon_sphere_centres[i] = image_coord_to_world(centre_l, centre_r)         
    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################
    # Computers error

    # Matches estimated sphere centres to the groundtruth sphere.
    # This can be reduced to the linear sum assignment problem.
    def match_estimates_to_groundtruths(truth, est):
        dist_pairs = np.zeros((len(truth), len(est)))
        for i in range(len(truth)):
            for j in range(len(est)):
                dist_pairs[i, j] = np.linalg.norm(truth[i] - est[j])
        # Computes which estimate corresponds to which groundtruth sphere
        row_indices, col_indices = linear_sum_assignment(dist_pairs)
        return row_indices, col_indices  

    truth = np.array(GT_cents)[:, :3] # Stores true sphere centres
    results_estimates.append(len(truth))
    results_estimates.append(len(recon_sphere_centres))
    # Prints if too many / too few or exatcly right number of sphere were detected
    if len(truth) > len(recon_sphere_centres):
        print("Too few spheres detected.", len(recon_sphere_centres), "/", len(truth), " detected.")
    elif len(truth) < len(recon_sphere_centres):
        print("Too many spheres detected.", len(recon_sphere_centres), "/", len(truth), " detected.")
    else:
        print("Detected correct number of spheres.", len(recon_sphere_centres), "/", len(truth), " detected.")

    # Match estimated centres to their corresponding ground truth spheres
    row_cor, col_cor = match_estimates_to_groundtruths(truth, recon_sphere_centres)
    truth_est_pairs = [(truth[i], recon_sphere_centres[j]) for i, j in zip(row_cor, col_cor)]
    rad_gt_ordered = [GT_rads[i] for i in row_cor]

    # Computes distance between centres
    dist = 0
    correct = 0
    thresh = 0.1
    for (t, e) in truth_est_pairs:
        # Computes distance
        d = np.linalg.norm(t - e)
        dist += d
        # Checks to see if distance is within thresh of radius in 3d
        rad_err = abs(t[1] * thresh)
        est_correct = True
        for i in range(0, 3):
            est_correct = est_correct and (t[i] - rad_err) <= e[i] and (t[i] + rad_err) >= e[i]
        if est_correct:
            correct += 1

        
    dist /= len(truth_est_pairs)
    print("Mean distance between centre estimate and truth:", dist)
    print("Total spheres correctly classified within", thresh * 100, "% of radius centre in each",
          "dimension: ", correct, "/", len(truth))
    results_estimates.append(dist)
    results_estimates.append(correct)


    # Creates two point clouds of sphere centres
    # Blue represents estimated sphere centres
    # Red represents the actual sphere centres
    if display_min:
        # Creates ground truth point cloud
        pcd_GTcents = o3d.geometry.PointCloud()
        pcd_GTcents.points = o3d.utility.Vector3dVector(truth)
        pcd_GTcents.paint_uniform_color([1., 0., 0.])
        # Creates reconstructed centres point cloud
        est_cents = o3d.geometry.PointCloud()
        est_cents.points = o3d.utility.Vector3dVector(recon_sphere_centres)
        est_cents.paint_uniform_color([0., 0., 1.])
        vis = o3d.visualization.Visualizer()
        # Draws plane and two point clouds
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents, est_cents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()
    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################
    # Estimates the radius of a circle defined in pixel coordinates given the corresponding world centre
    def estimate_radius(circ, centre_world, takenFromCam0):
        # Changes camera transfomrations depending on whether its from camera 0 or 1
        h_wc = H0_wc
        if not takenFromCam0: h_wc = H1_wc
        # Gets centre of circle
        (centre_x, centre_y), rad_circ = circ
        # Gets camera coords of centre point
        cent_point_cam = transform_points(np.array([centre_world]), h_wc)[0]
        d = cent_point_cam[2] # Depth is z in camera coords
        # Defines point on circumference and converts it to world coordinates.
        # This is achieved with perspective projection.
        circ_point_img = np.matmul(M, np.array([centre_x + rad_circ, centre_y, f]))
        circ_point_cam = (d / f) * circ_point_img
        circ_point_world = transform_points(np.array([circ_point_cam]), np.linalg.inv(h_wc))[0]
        # Computes rad
        rad = np.linalg.norm(circ_point_world - centre_world)
        return rad

    # For reference and view image calculates radius by projecting point on circumference
    # into world coords. Calculate distance between centre and point for radius. Takes average of
    # two images for radius.
    rads_est = []
    for i, (circ_l, circ_r) in enumerate(circ_pairs):
        # Computes radius estimate
        rad_l = estimate_radius(circ_l, recon_sphere_centres[i], True)
        rad_r = estimate_radius(circ_r, recon_sphere_centres[i], False)
        rad = (rad_l + rad_r) / 2.0
        rads_est.append(abs(rad))

    # Removes radius estimates that are not matched to corresponding circles 
    # (i.e. when we have more estimates than actual truth circles)
    rads_est = [rads_est[i] for i in range(0, len(rad_gt_ordered))]
    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################
    # Generates sphere point clouds to add to object mesh in open3d of provided spheres.
    def get_spheres(centers, radi, col):
        mesh = []
        for ((center_x, center_y, center_z), rad) in zip(centers, radi):
            sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=rad)
            sph_H = np.array(
                        [[1, 0, 0, center_x],
                        [0, 1, 0, center_y],
                        [0, 0, 1, center_z],
                        [0, 0, 0, 1]]
                    )
            sph_mesh.vertices = o3d.utility.Vector3dVector(
                transform_points(np.asarray(sph_mesh.vertices), sph_H)
            )
            # Draws sphere vertices in red
            sphere_cloud = o3d.geometry.PointCloud()
            sphere_cloud.points = o3d.utility.Vector3dVector(sph_mesh.vertices)
            sphere_cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array(col), (len(sph_mesh.vertices), 1)))
            mesh.append(sphere_cloud)
        return mesh

    # Displays estimated and true spheres
    def visualise_radi(sphere_centres, sphere_radi, true_cent, true_rad):
        K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

        objects_mesh = []
        # Adds plane
        objects_mesh.append(obj_meshes[0])
        # Adds estimated spheres to list as blue pointclouds
        objects_mesh += get_spheres(sphere_centres, sphere_radi, [0, 0, 1])
        # Adds true spheres to list as red pointclouds
        objects_mesh += get_spheres(true_cent, true_rad, [1, 0, 0])
          
        # Renders true spheres and reconstructed spheres.
        if display_min:
            cam = o3d.camera.PinholeCameraParameters()
            cam.intrinsic = K
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=img_width, height=img_height, left=0, top=0)
            for m in objects_mesh:
                vis.add_geometry(m)
            ctr = vis.get_view_control()
            render_list = [(H0_wc, 'Images/view0sphererad.png'), 
                        (H1_wc, 'Images/view1sphererad.png')]
            for (H_wc, name) in render_list:
                cam.extrinsic = H_wc
                ctr.convert_from_pinhole_camera_parameters(cam)
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(name, True)
            vis.run()
            vis.destroy_window()

    # Calcualtes error between estimates
    dist = 0
    thresh = 0.1
    correct = 0
    for (e, t) in zip(rads_est, rad_gt_ordered):
        d = np.linalg.norm(e -  t)
        dist += d
        if (t * (1.0 - thresh) <= e) and (t * (1.0 + thresh) >= e):
            correct += 1
    dist /= len(rads_est)
    print("Mean radius error:", dist)
    print("Total radi correctly caclulated within", thresh * 100, "% of true radius in each"
          , correct, "/", len(rad_gt_ordered))
    results_estimates.append(dist)
    results_estimates.append(correct)
    results_estimates.append(noise_param)
    # Visualises spheres
    visualise_radi(recon_sphere_centres, rads_est, truth, rad_gt_ordered)
    return (results_estimates, col_names)

    

if __name__ == '__main__':

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    # Visualise steps. If true, it visualises parts building up to final parts
    # (e.g. just circles, hough detection of circles, correspondence, epipolar lines).
    parser.add_argument('--additional_visualisation', dest='additional_visualisation', type=bool, default=True, 
                       help='Enables additional visualisations')
    # If noise expirements are enabled, impact of noise on performance is plotted.
    parser.add_argument('--noise_expirements', dest='noise_expirements', type=bool, default=False, 
                       help='Expirements with noise on relative pose.')

    args = parser.parse_args()

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(args.sph_sep_min,args.sph_sep_max,1)
        x = random.randrange(-h/2+2, h/2-2, step)
        z = random.randrange(-w/2+2, w/2-2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'Images/view0.png', 'Images/depth0.png'), 
                   (H1_wc, 'Images/view1.png', 'Images/depth1.png')]
    
    # Runs main reconstruction on scene.
    (_, col_names) = ipcv_part_2_main_function(args, "Images/", NO_NOISE, 0, True)
    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################
    # Generates matplotlib plots of relationship between noise and error
    def plot_results(results):
        # Creates plots to visualise results as noise changes
        # Plots noise factor against mean error
        plt.plot(results[:, 6], results[:, 2], label="Centre")
        plt.plot(results[:, 6], results[:, 4], label="Radius")
        plt.title("Impact of Relative Noise on Mean Reconstruction Error")
        plt.xlabel("Relative Noise Factor")
        plt.ylabel("Mean Reconstruction Error")
        plt.legend()
        plt.show()


    if args.noise_expirements:
        # Samples performance as noise varies and plots the result.
        print("Running code now WITH noise.")
        print("Gaussian addition noise.")
        results = []
        noise_factors_add = np.arange(0.00, 5.0, 0.1)
        for noise in noise_factors_add:
            print("Guassian noise with add: ", noise)
            (res, _) = ipcv_part_2_main_function(args, "Images/NoiseExpirements/GausAdd" + str(noise) + " ",
                                                     GAUSSIAN_ADD_NOISE, noise, False)
            results.append(res)
        print(col_names)
        print(results)
        plot_results(np.array(results))
    