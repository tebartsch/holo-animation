from typing import Callable

import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import comb


def smoothstep(x: np.ndarray, x_min=0, x_max=1, N=1) -> np.ndarray:
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def get_rectangles(grid_center: complex,
                   grid_width: float,
                   grid_height: float,
                   n_grid: int) -> np.ndarray[(None, None, None), np.complex128]:
    points_per_segment = 10

    x = np.linspace(-grid_width / 2, grid_width / 2,
                    n_grid, endpoint=False)
    y = np.linspace(-grid_height / 2, grid_height / 2,
                    n_grid, endpoint=False)

    rectangles = np.zeros((
        n_grid,
        n_grid,
        4 * (points_per_segment - 1) + 1),
        dtype=np.complex128)

    x_side = np.linspace(0, grid_width / n_grid, points_per_segment)
    x_box = np.concatenate((
        x_side,
        np.full(points_per_segment - 2, x_side[-1]),
        x_side[::-1],
        np.full(points_per_segment - 1, x_side[0])))
    y_side = np.linspace(0, grid_height / n_grid, points_per_segment)
    y_box = np.concatenate((
        np.full(points_per_segment - 1, y_side[0]),
        y_side,
        np.full(points_per_segment - 2, y_side[-1]),
        y_side[::-1]))

    for i, x_pos in enumerate(x):
        for j, y_pos in enumerate(y):
            pos = x_pos + 1j * y_pos
            box = x_box + 1j * y_box
            rectangles[i, j, :] = grid_center + pos + box

    return rectangles


def create_video(func: Callable[[np.complex128], np.complex128],
                 filename: str,
                 fps: int,
                 seconds: float,
                 grid_center: complex,
                 grid_width: float,
                 grid_height: float,
                 n_grid: int):
    """
    Create a video animating the functino `func` on a grid of complex numbers.

    :param func: Function which can be evaluated on numpy array of complex values
    :param filename: Location to stor output video
    :param fps: frames per second of the video
    :param seconds: total video time
    :param grid_center: the center of the grid which `func` is applied to
    :param grid_width: the width of the grid which `func` is applied to
    :param grid_height: the height of the grid which `func` is applied to
    :param n_grid: Number of grid cells in x- and y-dimension
    :return: None
    """

    # Video parameters
    frame_width = 1280
    frame_height = 720
    frame_ratio = frame_width / frame_height

    # Grid of rectangles to evaluate function on
    rectangles = get_rectangles(grid_center, grid_width, grid_height, n_grid)

    # Create array with start and end times between 0 (start of video) and 1 (end of video)
    # for every rectangle in the grid.
    n_grid_x = rectangles.shape[1]
    n_grid_y = rectangles.shape[0]
    start_times = np.linspace(0.1, 0.7, n_grid_x * n_grid_y).reshape(
        n_grid_y, n_grid_x)[::-1, :]
    end_times = start_times + 0.1

    # Get coordinates of enclosing rectangle.
    start_min_x, start_max_x = np.real(rectangles).min(), np.real(rectangles).max()
    start_min_y, start_max_y = np.imag(rectangles).min(), np.imag(rectangles).max()
    start_x_len = start_max_x - start_min_x
    start_y_len = start_max_y - start_min_y
    start_center = 0.5 * (start_min_x + start_max_x) + 0.5j * (start_min_y + start_max_y)

    # Get coordinates of enclosing rectangle of data transformed by `func`.
    transformed = func(rectangles)
    end_min_x, end_max_x = np.real(transformed).min(), np.real(transformed).max()
    end_min_y, end_max_y = np.imag(transformed).min(), np.imag(transformed).max()
    end_x_len = end_max_x - end_min_x
    end_y_len = end_max_y - end_min_y
    end_center = 0.5 * (end_min_x + end_max_x) + 0.5j * (end_min_y + end_max_y)

    # Offset of the rectangles at the beginning and the end of the video.
    x_len = start_x_len + end_x_len
    y_len = max(start_y_len, end_y_len)
    start_offset = - 0.5 * start_x_len / x_len
    end_offset = 0.5 * end_x_len / x_len

    scale = min(1 / (y_len * frame_ratio), 1 / x_len) * 0.98

    # The boundaries of the window of the complex plain which is shown in the video.
    window_bottom = -0.5 / frame_ratio
    window_top = 0.5 / frame_ratio
    window_left = - start_x_len / x_len
    window_right = end_x_len / x_len

    # Create the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"{filename}", fourcc, float(fps), (frame_width, frame_height))
    N = fps * seconds
    for k in tqdm(range(N)):
        # Create an array containing a value between 0 and 1 for every rectangle indicating how far it has moved
        # yet.
        time = k / (fps * seconds - 1)
        s = smoothstep((time - start_times) / (end_times - start_times), N=3)
        s[time < start_times] = 0
        s[time > end_times] = 1
        s = np.expand_dims(s, 2)

        # Apply func to the rectangles and move it according to the corresponding value in the array s.
        rectangles_mod = ((1 - s) * (scale * (rectangles - start_center) + start_offset)
                          + s * (scale * (func(rectangles) - end_center) + end_offset))

        def transform_x(x):
            return frame_width * ((x - window_left) / (window_right - window_left))

        def transform_y(y):
            return frame_height * ((window_top - y) / (window_top - window_bottom))

        # Obtain x_coordinates of (deformed) rectangles
        rectangles_x = transform_x(np.real(rectangles_mod)).astype(np.uint64)

        # Obtain y_coordinates of (deformed) rectangles
        rectangles_y = transform_y(np.imag(rectangles_mod)).astype(np.uint64)

        # Create frame with white background
        frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

        # draw red circle at center of start and end
        start_center_x = transform_x(np.real(start_offset)).astype(np.uint64)
        start_center_y = transform_y(np.imag(start_offset)).astype(np.uint64)
        cv2.circle(frame, [start_center_x, start_center_y],
                   radius=7,
                   color=(164, 163, 80),
                   thickness=-1,
                   lineType=cv2.LINE_AA)
        end_center_x = transform_x(np.real(scale*func(start_center) + end_offset)).astype(np.uint64)
        end_center_y = transform_y(np.imag(scale*func(start_center) + end_offset)).astype(np.uint64)
        cv2.circle(frame, [end_center_x, end_center_y],
                   radius=7,
                   color=(164, 163, 80),  # (56, 175, 252),
                   thickness=-1,
                   lineType=cv2.LINE_AA)

        # Add all deformed rectangles to a new frame
        m, n = rectangles.shape[:2]
        for i in range(m):
            for j in range(n):
                polygon = np.column_stack((
                    rectangles_x[i, j, :], rectangles_y[i, j, :]))
                cv2.polylines(frame, [polygon],
                              isClosed=False,
                              color=(0, 0, 0),
                              thickness=1,
                              lineType=cv2.LINE_AA  # cv2.LINE_AA stand for anti-aliased line
                              )

        video.write(frame)

    video.release()
