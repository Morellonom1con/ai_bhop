"""
Utilities for 2D interactive plotting
Used mainly in:
  - opentea / acquisition2D 

WARNING :  no not change the API unless you are also dealing with opentea

WARNING2 : We voluntarily avoid the use of Numpy here.
Indeed, The data is expected to be extremely small (dozens of points)
The cost of creating/ destroying numpy arrays is bigger than potential accelerations at this scale.

"""

from typing import List, Tuple

def point_in_line(line: List[List], curvi: float) -> Tuple[float, float]:
    """Return coordinates of a point in a line based on curvilinear abcissa"""
    if curvi < 0:
        return tuple(line[0])
    remaining_curvi = curvi

    i = 0
    while i < len(line) - 1:
        i += 1
        x0, y0 = line[i - 1]
        x1, y1 = line[i]
        lenght = hypot(x1 - x0, y1 - y0)

        if lenght < remaining_curvi:
            remaining_curvi -= lenght
            continue
        else:
            ratio = remaining_curvi / lenght
            return (1 - ratio) * x0 + ratio * x1, (1 - ratio) * y0 + ratio * y1

    # curvilinear abcissa is bigger than line
    return tuple(line[-1])


def link_two_lines(
    line_ext: List[List[float]], line_int: List[List[float]]
) -> Tuple[List[List[float]], List[List[float]], float]:
    """Add a mid point between the start of two lines to close the volume

    - each line is prepend by the same point
    - returns also the lenght of the added segment
    """

    mid_ptx = (line_ext[0][0] + line_int[0][0]) * 0.5
    mid_pty = (line_ext[0][1] + line_int[0][1]) * 0.5
    gamma = hypot(mid_ptx - line_ext[0][0], mid_pty - line_ext[0][1])
    line_ext_out = line_ext.copy()
    line_int_out = line_int.copy()
    line_ext_out.insert(0, [mid_ptx, mid_pty])
    line_int_out.insert(0, [mid_ptx, mid_pty])
    return line_ext_out, line_int_out, gamma


from typing import List, Optional, Tuple
from math import hypot


def is_near_point(
    x_1: float, y_1: float, x_2: float, y_2: float, tol: float = 0.2
) -> Optional[float]:
    """return distance if within the tolerance in viewport coords"""
    dist = hypot(x_1 - x_2, y_1 - y_2)
    return dist if dist < tol else None


def corner_coordinates(
    x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float, float, float]:
    """Return the minimum and maximum coordinates for the rectangle corners."""
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def coords_pix_2_real(
    x_pix: float,
    y_pix: float,
    calib_diag_pix: List[List[float]],
    calib_diag_real: List[List[float]],
) -> List[float]:
    """Return coords in real world, from wiewport coords"""
    ((x1_pix, y1_pix), (x2_pix, y2_pix)) = calib_diag_pix
    ((x1_real, y1_real), (x2_real, y2_real)) = calib_diag_real
    alphax = (x_pix - x1_pix) / (x2_pix - x1_pix)
    alphay = (y_pix - y1_pix) / (y2_pix - y1_pix)
    return [
        x1_real + alphax * (x2_real - x1_real),
        y1_real + alphay * (y2_real - y1_real),
    ]


def coords_real_2_pix(
    x_real: float,
    y_real: float,
    calib_diag_pix: List[List[float]],
    calib_diag_real: List[List[float]],
) -> List[float]:
    """Return coords in real world, from wiewport coords"""
    ((x1_pix, y1_pix), (x2_pix, y2_pix)) = calib_diag_pix
    ((x1_real, y1_real), (x2_real, y2_real)) = calib_diag_real
    alphax = (x_real - x1_real) / (x2_real - x1_real)
    alphay = (y_real - y1_real) / (y2_real - y1_real)
    return [x1_pix + alphax * (x2_pix - x1_pix), y1_pix + alphay * (y2_pix - y1_pix)]


def acq_lines_2_real(
    lines_list: List[List[List[float]]],
    calib_diag_pix: List[List[float]],
    calib_diag_real: List[List[float]],
):
    """Convert a  list of lines stored in pixels into real worlds coordinates"""
    all_lines = []
    for line_pix in lines_list:
        line_real = []
        for x_pix, y_pix in line_pix:
            line_real.append(
                coords_pix_2_real(x_pix, y_pix, calib_diag_pix, calib_diag_real)
            )
        all_lines.append(line_real)
    return all_lines


def acq_lines_2_pix(
    lines_list: List[List[List[float]]],
    calib_diag_pix: List[List[float]],
    calib_diag_real: List[List[float]],
):
    """Convert a  list of lines stored in pixels into real worlds coordinates"""
    all_lines = []
    for line_pix in lines_list:
        line_real = []
        for x_pix, y_pix in line_pix:
            line_real.append(
                coords_real_2_pix(x_pix, y_pix, calib_diag_pix, calib_diag_real)
            )
        all_lines.append(line_real)
    return all_lines


def find_line_from_point(
    lines_list: List[List[List[float]]], point: List[float]
) -> Optional[int]:
    """
    Return the index of the line containing the specified point.

    Parameters:
        lines_list (List[List[List[float]]]): List of lines, where each line is a list of points.
        point (List[float]): The point to search for.

    Returns:
        Optional[int]: The index of the line containing the point, or None if not found.
    """
    if point is None:
        return None

    for i, line in enumerate(lines_list):
        if point in line:
            return i
    return None


# =====================


def closest_line(lines, x, y) -> Tuple[int, int, float]:
    """return the closest line wr to a point, with the id of the closest point and the curvilineare abssica"""

    closest_dist = 10000000.0
    closest_line = None
    closest_point = None
    closest_curvi = None
    for i, line in enumerate(lines):
        point, dist, curvi = closest_segment_in_line(line, x, y)
        if dist < closest_dist:
            closest_line = i
            closest_point = point
            closest_curvi = curvi
            closest_dist = dist

    return closest_line, closest_point, closest_curvi


def closest_curvi_in_line(line, x, y) -> Tuple[int, int, float]:
    """return the closest line wr to a point, with the id of the closest point and the curvilineare abssica"""

    point, _, adim_curvi = closest_segment_in_line(line, x, y)
    curvi = 0
    if point >= 1:
        for i in range(point):
            x0, y0 = line[i]
            x1, y1 = line[i + 1]
            curvi += hypot(x1 - x0, y1 - y0)

    x0, y0 = line[point]
    x1, y1 = line[point + 1]
    curvi += hypot(x1 - x0, y1 - y0) * adim_curvi
    return curvi


def closest_segment_in_line(line, x, y) -> Tuple[int, float]:
    """Return the segment index closest to a point"""

    x0, y0 = line[0]
    closest_dist = 10000000.0
    closest_item = None
    closest_curvi = None
    for i in range(len(line) - 1):
        x1, y1 = line[i + 1]
        normal_dist, lateral_dist, curvi = dist_segment(x0, y0, x1, y1, x, y)
        dist = hypot(normal_dist, lateral_dist)
        if dist < closest_dist:
            closest_item = i
            closest_dist = dist
            closest_curvi = curvi
        x0, y0 = x1, y1

    return closest_item, closest_dist, closest_curvi


def dist_segment(
    x0: float, y0: float, x1: float, y1: float, x: float, y: float
) -> Tuple[float, float, float]:
    """
    Return the normal distance (distance to the line supporting the segment)
    the lateral distance (distance of the projection on the line to the center of the segment).
    and the curvilinear abscissa btw 0 and 1
    """
    # Compute the vector of the segment
    px, py = project_on_line(x0, y0, x1, y1, x, y)
    u = relative_position_segment(x0, y0, x1, y1, px, py)
    segment_len = hypot(x1 - x0, y1 - y0)
    if u < 0:
        lateral_dist = abs(u) * segment_len
        curvi = 0.0
    elif u < 1:
        lateral_dist = 0
        curvi = u
    else:
        lateral_dist = (u - 1) * segment_len
        curvi = 1.0

    normal_dist = hypot(px - x, py - y)
    return normal_dist, lateral_dist, curvi


def project_on_line(
    x0: float, y0: float, x1: float, y1: float, x: float, y: float
) -> Tuple[float, float]:
    """
    Project a point (x, y) onto a line defined by two points (x0, y0) and (x1, y1).
    Return the coordinates of the projection.
    """
    # Compute the vector of the line
    dx = x1 - x0
    dy = y1 - y0

    # Length of the line segment
    line_length_sq = dx**2 + dy**2

    # Edge case: If the two points are the same, the line is a single point
    if line_length_sq == 0:
        return x0, y0

    # Compute the vector from (x0, y0) to (x, y)
    vx = x - x0
    vy = y - y0

    # Projection factor (dot product divided by length squared)
    t = (vx * dx + vy * dy) / line_length_sq

    # Projection point (parameterized form)
    px = x0 + t * dx
    py = y0 + t * dy

    return px, py


def relative_position_segment(
    x0: float, y0: float, x1: float, y1: float, x: float, y: float
) -> float:
    """
    Return the relative position u of a point P(x, y) with respect to P0(x0, y0) and P1(x1, y1).

    Assumes P, P0, and P1 are collinear.

    Position:
      u = 0 if P = P0
      u = 1 if P = P1
      0 < u < 1 if P is between P0 and P1
      u < 0 if P is before P0
      u > 1 if P is after P1
    """
    # Compute the vector of the segment
    dx = x1 - x0
    dy = y1 - y0

    # Handle the edge case where the segment is a single point
    if dx == 0 and dy == 0:
        # If the segment is a point, u is defined to be 0 if P == P0, otherwise undefined
        return 0.0 if (x == x0 and y == y0) else float("nan")

    # Compute the vector from P0 to P
    vx = x - x0
    vy = y - y0

    # Compute the relative position u (projection factor along the segment's vector)
    # Avoid division by zero by ensuring either dx or dy is used
    if abs(dx) >= abs(dy):  # Use dx if it's more significant
        u = vx / dx
    else:  # Use dy if it's more significant
        u = vy / dy

    return u


def point_from_curvi(line, point_idx, curvi) -> Tuple[float, float, float]:
    """Return coordinates and curvilinear abscissa of a position on a line"""
    sum_curvi = 0
    for i in range(point_idx):
        (x0, y0) = line[i]
        (x1, y1) = line[i + 1]
        sum_curvi += hypot(x1 - x0, y1 - y0)
    (x0, y0) = line[point_idx]
    (x1, y1) = line[point_idx + 1]
    sum_curvi += hypot(x1 - x0, y1 - y0) * curvi

    x = (1 - curvi) * x0 + curvi * x1
    y = (1 - curvi) * y0 + curvi * y1
    return x, y, sum_curvi
