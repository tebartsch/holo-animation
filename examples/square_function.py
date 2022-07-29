import glob

from holo_animation import create_video
import cv2
from PIL import Image


def convert_mp4_to_jpgs(path):
    video_capture = cv2.VideoCapture(path)
    still_reading, image = video_capture.read()
    frame_count = 0
    while still_reading:
        cv2.imwrite(f"output/frame_{frame_count:03d}.jpg", image)
        still_reading, image = video_capture.read()
        frame_count += 1

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    return fps


def make_gif(frame_folder, fps):
    images = glob.glob(f"{frame_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save("square_function.gif", format="GIF", append_images=frames,
                   save_all=True, duration=1000/fps, loop=0)


def main():
    create_video(lambda z: z**2, './square-function.mp4',
                 grid_center=0 + 0j,
                 grid_width=1,
                 grid_height=1,
                 n_grid=30,
                 fps=50,
                 seconds=5)
    fps = convert_mp4_to_jpgs('./square-function.mp4')
    make_gif("output", fps)


if __name__ == "__main__":
    main()
