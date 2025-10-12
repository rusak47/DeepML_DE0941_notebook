import torch
import numpy as np
import sys
import matplotlib
from sympy.codegen.rewriting import powm1_opt

if sys.platform == 'darwin':
    matplotlib.use("MacOSX") # for mac
else:
    matplotlib.use("TkAgg") # for unix/windows

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7, 7) # size of window
plt.ion()
plt.style.use('dark_background')

y = np.array([-3.0, 0])
x = np.array([0, 0])

USE_MAE = False
is_running = True
def button_press_event(event):
    global y
    y = np.array([event.xdata, event.ydata])

def press(event):
    global is_running
    print('press', event.key)
    if event.key == 'escape':
        is_running = False # quits app
        plt.close('all')

def on_close(event):
    global is_running
    is_running = False

fig, _ = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('button_press_event', button_press_event)
fig.canvas.mpl_connect('key_press_event', press)

length1 = 2.0
length2 = 2.0
length3 = 1.0
ROBOT_ARM_TOTAL_LENGTH = length1 + length2 + length3
theta_1 = np.deg2rad(-10)
theta_2 = np.deg2rad(-10)
theta_3 = np.deg2rad(-10)

"""
This is the standard 2D rotation matrix for counterclockwise rotation:

R(θ)=[cos(θ) sin(θ)
    −sin(θ)cos(θ)]
    
This matrix rotates a vector (x, y) counterclockwise by theta radians around the origin (0, 0).

Copied from: ChatGPT - <https://chatgpt.com/?utm_source=google&utm_medium=paidsearch_brand&utm_campaign=GOOG_C_SEM_GBR_Core_CHT_BAU_ACQ_PER_MIX_ALL_EMEA_LV_LV_050725&utm_term=chatgpt&utm_content=180229121425&utm_ad=757793550445&utm_match=e&gad_source=1&gad_campaignid=22532538619&gclid=CjwKCAjwuePGBhBZEiwAIGCVS5Az3HAtnGMXLBNRzOvYRWorVrfRoLMiQbKM_6SjnIzP1Hecyx4DABoChr8QAvD_BwE>
"""
def rotation(theta):
    radians = theta #np.deg2rad(theta)
    R = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    return R

def d_rotation(theta):
    radians = theta #np.deg2rad(theta)
    R = np.array([
        [-np.sin(radians), -np.cos(radians)],
        [np.cos(radians), -np.sin(radians)]
    ])
    return R

def dd_rotation(theta):
    radians = theta #np.deg2rad(theta)
    R = np.array([
        [-np.cos(radians), np.sin(radians)],
        [-np.sin(radians), -np.cos(radians)]
    ])
    return R


"""
    get angle between p and it's projection on x axis

    cos(theta) =         (v1@v2)
                    -----------------
                     ||v1|| * ||v2||
                     where || v1 || - magnitude

"""
# todo pass projection points as arguments to test with next segments
def angle_between_projection(p0, p1):
    p_proj_0 = np.array([p0[0], 0])
    p_proj_1 = np.array([p1[0], 0])
    v1 = p1 - p0
    v2 = p_proj_1 - p_proj_0

    v1_mag = np.sqrt(v1[0] ** 2 + v1[1] ** 2)  # magnitude of vector aka length (scalar)
    v2_mag = np.sqrt(v1[0] ** 2 + v1[1] ** 2)

    theta = np.arccos((v1 @ v2) / (v1_mag * v2_mag))
    return theta


"""
    return 0 ??? todo When we say:
"Red dot is exactly perpendicular to the tangent at the end"
We mean:
The vector from the line's end (p3) to the red dot (x - p3)
Is perpendicular to the direction vector of the line (tangent = p3 - p2)
Copied from: ChatGPT - <https://chatgpt.com/>

            -1 if y below
            1 if y above
"""
def signed_projection(last_segment_p1, last_segment_p2, y):
    tan_dir = last_segment_p2 - last_segment_p1 # tangent direction at the last segment
    vec_to_y = y - last_segment_p2 # vector from last segment point to y
    tan_dir_mag = np.sqrt(last_segment_p2[0]**2 + last_segment_p2[1]**2) # magnitude of vector aka length (scalar)
    #normalize direction vector: keep direction, but produce unit vector (len=1)
    norm_tan_dir = tan_dir / tan_dir_mag
    print(f">> norm_tan_dir: {norm_tan_dir}") # abs values below 1
    print(f">> proj:{vec_to_y@norm_tan_dir}")
    return vec_to_y@norm_tan_dir

def is_parallel_segments(p0, p1, p2):
    v1 = p0 - p1 # direction vector
    v2 = p1 - p2 # both segment direction vectors face in one direction when their cross product is zero
    cross = np.cross(v1,v2)
    print(f"cross: {cross:.3f}")
    #return np.isclose(np.abs(cross),1e-1)
    return np.abs(cross) < 2e-1

# f(theta,x,t) = R(theta)*x+t
def f(theta,x,t):
    rotated = rotation(theta)
    dot = np.dot(rotated, x) #shortcut via @, like A@x
    dot += t
    return dot

#def loss(y,p):

v_length3 = 0
while is_running:
    plt.clf()

    t1 = np.array([0.0, 1.0]) * length1
    t2 = np.array([0.0, 1.0]) * length2
    t3 = np.array([0.0, 1.0]) * (length3+v_length3) #change robot length

    _r1 = rotation(theta_1)
    _r2 = rotation(theta_2)
    _r3 = rotation(theta_3)

    d_r1 = d_rotation(theta_1)
    d_r2 = d_rotation(theta_2)
    d_r3 = d_rotation(theta_3)

    #f(theta_1, [1,1], t)

    # Each position (p1, p2, p3) builds on the previous one, applying cumulative rotations.
    p1 = _r1 @ t1 # Rotate t1 around the origin using _r1 — this gives the position of the first point in global frame

    # calculate angle relative to x, when it's greater than abs 90 error should increase
    t1_angle = angle_between_projection(x, p1)
    print(f"angle: {np.rad2deg(t1_angle)}")  # 90* perpendicular, 0* parallel, so we want angle to be above 0

    p2 = p1 + _r1 @ (_r2 @ t2) # _r1@t1 + _r1 @ (_r2 @ t2)
                               # Rotate t2 by _r2 (local frame), then by _r1 (parent frame), then translate by p1 — gives the second point
    p3 = p2 + _r1 @ ( _r2 @ ( _r3 @ t3 ) ) # _r1@t1 + _r1 @ (_r2 @ t2) + _r1 @ ( _r2 @ ( _r3 @ t3 ) )
                                           # Rotate t3 by _r3 (local), then by _r2 (parent), then _r1 (global), add to p2 — gives the third point
    points = []

    arm_curr_distance_to_ground = np.sqrt(np.sum((x - p3) ** 2))
    arm_curr_distance_to_target = np.sqrt(np.sum((p3 - y) ** 2))
    target_distance_to_ground = y[1]

    direction_projection = signed_projection(p2, p3, y) # last segment direction projection relative to red target (y)
    if direction_projection > 1e-2:
        pos_title = "above"
    elif direction_projection < -1e-2:
        pos_title = "below"
    else:
        pos_title = "close"

    l_eps = 1e-3
    l_alph = 1e-1 # speed of shrinking/extending arm
    #if arm_curr_distance_to_ground <= ROBOT_ARM_TOTAL_LENGTH and v_length3 > 0:
    if ROBOT_ARM_TOTAL_LENGTH - arm_curr_distance_to_ground >= l_eps  and v_length3 > 0:
        v_length3 -= l_alph  # shrink arm to default size

    #if current distance to ground is greater than robot default length and arm last segment distance to target is greater than error, then extend arm last segment
    elif ROBOT_ARM_TOTAL_LENGTH - arm_curr_distance_to_ground < l_eps and arm_curr_distance_to_target > 1e-1:
        if "above" == pos_title:
            v_length3 += l_alph
        elif "below" == pos_title:
            v_length3 -= l_alph  # extend arm to reach target

    if v_length3 < 0: # correct overshrinking (negative length)
        v_length3 = 0
    """
    if distance_to_ground <= total_length and v_length3 > 0:
        v_length3 -= l_alph #shrink arm to default size
    elif total_length - distance_to_ground < l_eps and distance_to_target > 1e-1: #todo identify when to stop; y <- press event
        #todo if loss increases stop extending
        #todo identify arm segments position relative to each other: extend only if they are parallel
        seg12_parallel = is_parallel_segments(x,p1,p2)
        seg23_parallel = is_parallel_segments(p1, p2, p3)
        print(f"parall: {seg12_parallel} {seg23_parallel}")
        if seg12_parallel and seg23_parallel:
            v_length3 += l_alph #extend arm to reach target
        #elif v_length3 > 0:
        #    v_length3 -= l_alph
    """

    #loss = (p3 - p2) @ (p3 -p2)
    #loss = (y[0]-p2[0]) * (y[0]-p2[0]) + (y[1]-p2[1])*(y[1]-p2[1])
    #loss = np.sum((y-p2)**) #2segment
    if USE_MAE:
        loss = np.sum(np.abs(y-p3)) # MAE
    else:
        loss = np.sum((y-p3)**2) #MSE sum 1 to n (y-p3)^2

    points.append(x) #p0
    #points.append(np.array(x) + t)
    points.append(p1)
    points.append(p2)
    points.append(p3)

    np_points = np.array(points)

    plt.title(f"""loss: {loss:.3f};
                    \n theta_1: {round(np.rad2deg(theta_1))}* theta_2: {round(np.rad2deg(theta_2))}* theta_3: {round(np.rad2deg(theta_3))}* 
                    \n p0-p3_dist: {arm_curr_distance_to_ground:.3f} l3: {length3 + v_length3:.3f}; dot_pos: {pos_title}; dot_distance: {target_distance_to_ground:.3f}""", linespacing=0.4)

    if len(np_points):
        plt.plot(np_points[:, 0], np_points[:, 1])
    plt.scatter(y[0], y[1], s=50, c='r')

    plt.xlim(-5, 5)
    #plt.ylim(0, 10)
    plt.ylim(-4, 10) # show negative part to test border crossing
    plt.draw()
    plt.pause(1e-3)


    if USE_MAE:
        # differentiate MAE: for the derivative of an absolute value function: d/dx |u(x)| = u(x)/|u(x)| * u'(x)
        # simplified: use the signum function form: sgn(u(x)) * u'(x).
        d_loss_y = -np.sign(y-p3)
    else:
        #  the derivative of loss with respect to the targets p
        #d_loss_y_scal = -2 * (y[0] - p2[0])  - 2 * (y[1] - p2[1]) # scalar
        #loss_y = np.sum(-2*(y-p2)) # same as above yet shorter
        #d_loss_y = -2*(y-p2) # for 2 segment arm
        d_loss_y = -2 * (y - p3) # for 3 seg arm

    ##p1: _r1@t1
    ##p2: _r1@t1 + _r1 @ (_r2 @ t2)
    ##p3: _r1@t1 + _r1 @ (_r2 @ t2) + _r1 @ ( _r2 @ ( _r3 @ t3 ) )
    ##
    #d_theta_1_scalar = np.sum(d_loss_y_scal * d_r1 @ t + d_r1 @ _r2 @ t )
    #d_theta_1_2seg = np.sum(-2*(y-p2)  @ (d_r1 @ t1 + d_r1 @ _r2 @ t1) )
    # 0* - perpendicular to x axis
    """
            dLmse ~= some func * d theta_1, where dLmse is func area and d theta_1 some step; some func is a func value at step d theta_1
            
          dLmse
        --------  ~ some func
        d theta_1
    """
    d_theta_1 = np.sum(d_loss_y  @ (d_r1 @ t1 + d_r1 @ _r2 @ t2 + d_r1 @ _r2 @ _r3 @ t3) )

    #d_theta_2_scalar = np.sum(d_loss_y_scal * _r1 @ d_r2 @ t2) # it seems like mathematically equivalent, yet this one causes drifts
                                                    # too early flattening to scalar value loses precision of calculations
    #d_theta_2 = np.sum(-2*(y-p2) @ _r1 @ d_r2 @ t2 ) # 2 seg arm
    # 90* - perpendicular to segment 1, yet angle calculated from external side (adjacent angle)
    d_theta_2 = np.sum(d_loss_y @ (_r1 @ d_r2 @ t2 + _r1 @ d_r2 @ _r3 @ t3) )

    # 90* - perpendicular to segment 2, yet angle calculated from external side
    d_theta_3 = np.sum(d_loss_y @ _r1 @ _r2 @ d_r3 @ t3 )

    alpha = 5e-3
    eps = 1e-3
    beta = 50 # exponential speedup constant for fading error coefficient; setting 1 makes change veeery slow
    if abs(loss) > eps:#todo dont allow to increase loss value -> predict next?
        print(f"loss {loss}")
        print(f"distance: {arm_curr_distance_to_ground:.5f}; {arm_curr_distance_to_target:.5f}")
        #theta-= loss_y*alpha#*d_theta_2 #stochastic gradient descent SGD
        """
            Gradient Descent - a method for finding a minimum of the function (loss fn in ML)
                by iteratively moving in the direction of the negative gradient
            in std gradient descent the gradient is computed all the data;
            in stochastic only one data point is used, which introduces randomness (hence the name)
        """

        ## After fiddling with it (1dt segment angle relative to x axis) for a while,
        # I've come up with the idea to use an exponential function as a fading coefficient.
        #
        t1_angle_norm = t1_angle/90 # 1 segment angle changes from 0 to 90 and back to 0, ideal to stop from crossing x axis
                                    # use normalized value in the range [0,1]
        exp_slope_coeff = 1 # control how fast exponential value changes (1 for near to straight line)
        fading_coeff_fn = ((np.exp(exp_slope_coeff*t1_angle_norm) -1)/(np.exp(exp_slope_coeff) - 1)) * beta
        theta_1 -= d_theta_1 * alpha * fading_coeff_fn
        print(f"-----------------")
        print(f"origin: {(d_theta_1 * alpha):.4f}; new: {(d_theta_1 * alpha * fading_coeff_fn):.4f}; e:{fading_coeff_fn:.4f}")
        print(f"y.x: {y[1]}; y_norm: {y[1] / 10}")
        print(f"-----------------")

        if v_length3 <= 0:
            # reaches 1st segment at angles abs(180)
            theta_2 -= d_theta_2 * alpha
            theta2_norm
            theta_3 -= d_theta_3 * alpha


    """
    TODO Avoid having the arm cross over itself or drop below level 0 
    (this can be achieved by:
      - adding extra error and 
        # loss fn: if first segment angle relative to x is near to 0 - disable angle correction
      - additional dTheta from different points on the arm's central parts <- which ones - segments? segment midpoints?
      - and you can also perform additional maximization - ?
    NB NO IFs!!!)
    Copied from: 2025-Q3-M-AI-RTU: 3.7. Homework - Implement a robotic arm with 3 segments | YellowRobot.xyz - <https://moodle.yellowrobot.xyz/mod/assign/view.php?id=316>"""


