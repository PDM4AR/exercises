from typing import Sequence

from dg_commons import SE2Transform

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points

import numpy as np


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    # TODO implement here your solution
    min_radius = wheel_base/np.tan(max_steering_angle)
    return DubinsParam(min_radius=min_radius)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    # TODO implement here your solution
    position = current_config.p
    theta = current_config.theta
    r_center = np.array([radius*np.cos(theta-np.pi/2), radius*np.sin(theta-np.pi/2)]) + position
    l_center = np.array([radius*np.cos(theta+np.pi/2), radius*np.sin(theta+np.pi/2)]) + position
    right_circle = Curve.create_circle(center=SE2Transform(r_center, theta=0), config_on_circle=current_config,
                                       radius=radius, curve_type=DubinsSegmentType.RIGHT) 
    left_circle = Curve.create_circle(center=SE2Transform(l_center, theta=0), config_on_circle=current_config,
                                       radius=radius, curve_type=DubinsSegmentType.LEFT) 
    return TurningCircle(left=left_circle, right=right_circle)

# def get_circle_type(circle: Curve):
#     center_p = circle.center.p
#     config_p = circle.start_config.p
#     config_theta = circle.start_config.theta
#     return "counter-clock" if np.cross(config_p - center_p, [np.cos(config_theta), np.sin(config_theta)]) > 0 else "clockwise"
def calculate_circle_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    center_start = circle_start.center.p
    center_end = circle_end.center.p
    center_vec = center_end - center_start
    d = np.linalg.norm(center_vec)
    radius = circle_start.radius

    type_start = circle_start.type
    type_end = circle_end.type
    
    if d > 4*radius:
        return []
    
    if type_start == DubinsSegmentType.LEFT and type_end == DubinsSegmentType.RIGHT:
        return []
   
    if type_start == DubinsSegmentType.RIGHT and type_end == DubinsSegmentType.LEFT:
        return []
    
    if type_start == DubinsSegmentType.LEFT and type_end == DubinsSegmentType.LEFT:
        theta_r = np.arccos((d / 2)/(2*radius))
        offset_r = np.pi/2
        offset_p = 1
    if type_start == DubinsSegmentType.RIGHT and type_end == DubinsSegmentType.RIGHT:
        theta_r = np.arccos((d / 2)/(2*radius))
        offset_r = -np.pi/2
        offset_p = -1
   
        
    c, s = np.cos(theta_r), np.sin(theta_r)
    R = np.array(((c, -s), (s, c))) # rotation matrix
    n_rad = np.dot(R, center_vec / d * radius) # vector from the center to the tagent point
    p_start = center_start + n_rad  # tagent point (start)
    theta_start = np.arctan2(n_rad[1], n_rad[0]) + offset_r

    p_center = center_start + 2*n_rad
    
    p_end = (p_center +  center_end)/2   
    arc_angle = np.pi+2*offset_p*theta_r
    theta_end = theta_start - offset_p*arc_angle
    
    curve_1 =  Curve(center=SE2Transform(p_center,0), start_config=SE2Transform(p_start, theta_start), end_config=SE2Transform(p_end, theta_end), radius=radius,
                    arc_angle=arc_angle, curve_type= DubinsSegmentType.LEFT if circle_start.type==DubinsSegmentType.RIGHT else DubinsSegmentType.RIGHT)
    
    c, s = np.cos(-theta_r), np.sin(-theta_r)
    R = np.array(((c, -s), (s, c))) # rotation matrix
    
    n_rad = np.dot(R, center_vec / d * radius) # vector from the center to the tagent point
    
    p_start = center_start + n_rad  # tagent point (start)
    
    theta_start = np.arctan2(n_rad[1], n_rad[0]) + offset_r
    
    p_center = center_start + 2*n_rad
    
    p_end = (p_center +  center_end)/2   
    arc_angle = np.pi-2*offset_p*theta_r
    theta_end = theta_start - offset_p*arc_angle
    
    curve_2 =  Curve(center=SE2Transform(p_center,0), start_config=SE2Transform(p_start, theta_start), end_config=SE2Transform(p_end, theta_end), radius=radius,
                    arc_angle=arc_angle, curve_type= DubinsSegmentType.LEFT if circle_start.type==DubinsSegmentType.RIGHT else DubinsSegmentType.RIGHT)
    
    return [curve_1, curve_2]

def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    center_start = circle_start.center.p
    center_end = circle_end.center.p
    center_vec = center_end - center_start
    d = np.linalg.norm(center_vec)
    radius = circle_start.radius

    type_start = circle_start.type
    type_end = circle_end.type
    
    if type_start == DubinsSegmentType.LEFT and type_end == DubinsSegmentType.RIGHT:
        if d < 2*radius:
            return []
        theta_r = -np.arccos(radius / (d / 2))
        offset_r = np.pi/2
        offset_p = -1
    if type_start == DubinsSegmentType.RIGHT and type_end == DubinsSegmentType.LEFT:
        if d < 2*radius:
            return []
        theta_r = np.arccos(radius / (d / 2))
        offset_r = -np.pi/2
        offset_p = -1
    if type_start == DubinsSegmentType.LEFT and type_end == DubinsSegmentType.LEFT:
        theta_r = -np.pi/2
        offset_r = np.pi/2
        offset_p = 1
    if type_start == DubinsSegmentType.RIGHT and type_end == DubinsSegmentType.RIGHT:
        theta_r = np.pi/2
        offset_r = -np.pi/2
        offset_p = 1
        
    c, s = np.cos(theta_r), np.sin(theta_r)
    R = np.array(((c, -s), (s, c))) # rotation matrix
    
    n_rad = np.dot(R, center_vec / d * radius) # vector from the center to the tagent point
    
    p_start = center_start + n_rad  # tagent point (start)
    
    theta_start = np.arctan2(n_rad[1], n_rad[0]) + offset_r
    
    p_end = center_end + offset_p*n_rad    # tagent point (end)
    theta_end = theta_start       # the orientation is the same
    
    tagent_line = Line(start_config=SE2Transform(p_start, theta_start), end_config=SE2Transform(p_end, theta_end))
    
    return [tagent_line]

def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins path!
    start_circles = calculate_turning_circles(start_config, radius)
    end_circles = calculate_turning_circles(end_config, radius)

    shortest_path = None
    shortest_length = float('inf')

    for start_circle in [start_circles.left, start_circles.right]:
        for end_circle in [end_circles.left, end_circles.right]:
            tangents = calculate_tangent_btw_circles(start_circle, end_circle)
            circles_curves = calculate_circle_btw_circles(start_circle, end_circle)
            
            if circles_curves:
                for mid_circle in circles_curves:
                    if start_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_start = mid_circle.start_config.theta - start_circle.start_config.theta
                        arc_angle_end = end_circle.end_config.theta - mid_circle.end_config.theta
                        
                    elif start_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_start = start_circle.start_config.theta - mid_circle.start_config.theta
                        arc_angle_end = mid_circle.end_config.theta - end_circle.end_config.theta
                        
                    curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=mid_circle.start_config, radius=radius,
                    arc_angle=arc_angle_start, curve_type=start_circle.type)
                    curve_end = Curve(center=end_circle.center, start_config=mid_circle.end_config, end_config=end_circle.start_config, radius=radius,
                    arc_angle=arc_angle_end, curve_type=end_circle.type)
                    length = curve_start.length + mid_circle.length + curve_end.length

                    # Update the shortest path and its length if necessary
                    if length < shortest_length:
                        shortest_length = length
                        shortest_path = [curve_start, mid_circle, curve_end]
                    
            if tangents:
                # Loop over all tangent lines
                for tangent in tangents:
                    if start_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_start = tangent.start_config.theta-start_circle.start_config.theta
                    elif start_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_start = start_circle.start_config.theta - tangent.start_config.theta
                    if end_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_end = end_circle.end_config.theta - tangent.end_config.theta
                    elif end_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_end = tangent.end_config.theta - end_circle.end_config.theta
                    curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=tangent.start_config, radius=radius,
                    arc_angle=arc_angle_start, curve_type=start_circle.type)
                    curve_end = Curve(center=end_circle.center, start_config=tangent.end_config, end_config=end_circle.start_config, radius=radius,
                    arc_angle=arc_angle_end, curve_type=end_circle.type)
                    length = curve_start.length + tangent.length + curve_end.length

                    # Update the shortest path and its length if necessary
                    if length < shortest_length:
                        shortest_length = length
                        shortest_path = [curve_start, tangent, curve_end]

    return shortest_path
    
    # return []  # e.g., [Curve(), Line(),..]


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    # TODO implement here your solution
    # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
    # Turning circles for the start and end configurations
    
    start_circles = calculate_turning_circles(start_config, radius)
    end_circles = calculate_turning_circles(end_config, radius)

    shortest_forward_path = None
    shortest_forward_length = float('inf')
    
    # For forward
    for start_circle in [start_circles.left, start_circles.right]:
        for end_circle in [end_circles.left, end_circles.right]:
            tangents = calculate_tangent_btw_circles(start_circle, end_circle)
            circles_curves = calculate_circle_btw_circles(start_circle, end_circle)
            
            if circles_curves:
                for mid_circle in circles_curves:
                    if start_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_start = mid_circle.start_config.theta - start_circle.start_config.theta
                        arc_angle_end = end_circle.end_config.theta - mid_circle.end_config.theta
                        
                    elif start_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_start = start_circle.start_config.theta - mid_circle.start_config.theta
                        arc_angle_end = mid_circle.end_config.theta - end_circle.end_config.theta
                        
                    curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=mid_circle.start_config, radius=radius,
                    arc_angle=arc_angle_start, curve_type=start_circle.type)
                    curve_end = Curve(center=end_circle.center, start_config=mid_circle.end_config, end_config=end_circle.start_config, radius=radius,
                    arc_angle=arc_angle_end, curve_type=end_circle.type)
                    length = curve_start.length + mid_circle.length + curve_end.length

                    # Update the shortest path and its length if necessary
                    if length < shortest_forward_length:
                        shortest_forward_length = length
                        shortest_forward_path = [curve_start, mid_circle, curve_end]
                    
            if tangents:
                # Loop over all tangent lines
                for tangent in tangents:
                    if start_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_start = tangent.start_config.theta-start_circle.start_config.theta
                    elif start_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_start = start_circle.start_config.theta - tangent.start_config.theta
                    if end_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_end = end_circle.end_config.theta - tangent.end_config.theta
                    elif end_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_end = tangent.end_config.theta - end_circle.end_config.theta
                    curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=tangent.start_config, radius=radius,
                    arc_angle=arc_angle_start, curve_type=start_circle.type)
                    curve_end = Curve(center=end_circle.center, start_config=tangent.end_config, end_config=end_circle.start_config, radius=radius,
                    arc_angle=arc_angle_end, curve_type=end_circle.type)
                    length = curve_start.length + tangent.length + curve_end.length

                   # Update the shortest path and its length if necessary
                    if length < shortest_forward_length:
                        shortest_forward_length = length
                        shortest_forward_path = [curve_start, tangent, curve_end]
                        
    # For backward
    start_circles = calculate_turning_circles(end_config, radius)
    end_circles = calculate_turning_circles(start_config, radius)
    
    shortest_backward_path = None
    shortest_backward_length = float('inf')
    
    for start_circle in [start_circles.left, start_circles.right]:
        for end_circle in [end_circles.left, end_circles.right]:
            tangents = calculate_tangent_btw_circles(start_circle, end_circle)
            
            # if circles_curves:
            #     for mid_circle in circles_curves:
            #         if start_circle.type == DubinsSegmentType.LEFT:
            #             arc_angle_start = mid_circle.start_config.theta - start_circle.start_config.theta
            #             arc_angle_end = end_circle.end_config.theta - mid_circle.end_config.theta
                        
            #         elif start_circle.type == DubinsSegmentType.RIGHT:
            #             arc_angle_start = start_circle.start_config.theta - mid_circle.start_config.theta
            #             arc_angle_end = mid_circle.end_config.theta - end_circle.end_config.theta
                        
            #         curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=mid_circle.start_config, radius=radius,
            #         arc_angle=arc_angle_start, curve_type=start_circle.type, gear=Gear.REVERSE)
            #         curve_end = Curve(center=end_circle.center, start_config=mid_circle.end_config, end_config=end_circle.start_config, radius=radius,
            #         arc_angle=arc_angle_end, curve_type=end_circle.type, gear=Gear.REVERSE)
            #         length = curve_start.length + mid_circle.length + curve_end.length
                    
            #         mid_circle.gear = Gear.REVERSE

            #         # Update the shortest path and its length if necessary
            #         if length < shortest_backward_length:
            #             shortest_backward_length = length
            #             shortest_backward_path = [curve_end, mid_circle, curve_start]
            
            if tangents:
                # Loop over all tangent lines
                for tangent in tangents:
                    if start_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_start = tangent.start_config.theta-start_circle.start_config.theta
                    elif start_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_start = start_circle.start_config.theta - tangent.start_config.theta
                    if end_circle.type == DubinsSegmentType.LEFT:
                        arc_angle_end = end_circle.end_config.theta - tangent.end_config.theta
                    elif end_circle.type == DubinsSegmentType.RIGHT:
                        arc_angle_end = tangent.end_config.theta - end_circle.end_config.theta
                    # curve_start = Curve(center=start_circle.center, start_config=start_circle.start_config, end_config=tangent.start_config, radius=radius,
                    # arc_angle=arc_angle_start, curve_type=start_circle.type)
                    # curve_end = Curve(center=end_circle.center, start_config=tangent.end_config, end_config=end_circle.end_config, radius=radius,
                    # arc_angle=arc_angle_end, curve_type=end_circle.type)
                    # length = curve_start.length + tangent.length + curve_end.length
                    # print('length',tangent.length)
                    
                    # Inverse
                    back_curve_start = Curve(center=end_circle.center, start_config=end_circle.end_config, end_config=tangent.end_config, radius=radius,
                    arc_angle= arc_angle_end, curve_type= end_circle.type, gear = Gear.REVERSE)
                    back_curve_end = Curve(center=start_circle.center, start_config=tangent.start_config, end_config=start_circle.start_config, radius=radius,
                    arc_angle= arc_angle_start, curve_type= start_circle.type, gear = Gear.REVERSE)                    
                    
                    backward_tagent = Line(start_config=tangent.end_config, end_config=tangent.start_config, gear= Gear.REVERSE)
                    length = back_curve_start.length + backward_tagent.length + back_curve_end.length
                    
                    # Update the shortest path and its length if necessary
                    if length < shortest_backward_length:
                        shortest_backward_length = length
                        shortest_backward_path = [back_curve_start, backward_tagent, back_curve_end]
     
    shortest_path = shortest_forward_path if shortest_forward_length < shortest_backward_length else shortest_backward_path
     
    return shortest_path

# from typing import Sequence

# from dg_commons import SE2Transform

# from pdm4ar.exercises.ex05.structures import *
# from pdm4ar.exercises_def.ex05.utils import extract_path_points


# class PathPlanner(ABC):
#     @abstractmethod
#     def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
#         pass


# class Dubins(PathPlanner):
#     def __init__(self, params: DubinsParam):
#         self.params = params

#     def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
#         """ Generates an optimal Dubins path between start and end configuration

#         :param start: the start configuration of the car (x,y,theta)
#         :param end: the end configuration of the car (x,y,theta)

#         :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
#         """
#         path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
#         se2_list = extract_path_points(path)
#         return se2_list


# class ReedsShepp(PathPlanner):
#     def __init__(self, params: DubinsParam):
#         self.params = params

#     def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
#         """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

#         :param start: the start configuration of the car (x,y,theta)
#         :param end: the end configuration of the car (x,y,theta)

#         :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
#         """
#         path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
#         se2_list = extract_path_points(path)
#         return se2_list


# def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
#     # TODO implement here your solution
#     return DubinsParam(min_radius=0)


# def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
#     # TODO implement here your solution
#     dummy_circle = Curve.create_circle(center=SE2Transform.identity(), config_on_circle=SE2Transform.identity(),
#                                        radius=0.1, curve_type=DubinsSegmentType.LEFT)  # TODO remove
#     return TurningCircle(left=dummy_circle, right=dummy_circle)


# def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
#     # TODO implement here your solution
#     return []  # i.e., [Line(),...]


# def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
#     # TODO implement here your solution
#     # Please keep segments with zero length in the return list & return a valid dubins path!
#     return []  # e.g., [Curve(), Line(),..]


# def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
#     # TODO implement here your solution
#     # Please keep segments with zero length in the return list & return a valid dubins/reeds path!
#     return []  # e.g., [Curve(..,gear=Gear.REVERSE), Curve(),..]
