from __future__ import annotations

import glob
import os
import subprocess
from typing import Callable
import skvideo.io  

import cv2
import ffmpeg
import imageio
import numpy as np
from tqdm import tqdm, trange

class ZoomOut:
    def __init__(
        self,
        videos_dir: str,
        preprocess_files_fn: Callable,
        load_video_fn: Callable,
        # === gallery overview ===
        rows: int = 5,
        cols: int = 5,
        num_start_rows: int = 2,
        num_start_cols: int = 2,
        mid_pause: bool = False,
        r_mid: int = 5,
        c_mid: int = 5,
        # === video info ===
        input_resolution: tuple[int, int] = (540, 960),
        num_frames: int = 480*2,
        fps: int = 30,
        margin_h: int = 6,
        margin_w: int = 6,
        final_resolution: tuple[int, int] = (1440, 2560),
        zoom_out_end_sec: int = 0,
        # === other ===
        num_processes: int | None = 60,
        output_dir: str = ".",
        output_name: str = "gallery",
    ):
        vars(self).update((k, v) for k, v in vars().items() if k != "self")
        self.preprocess_files_fn = preprocess_files_fn or os.path.listdir
        self.num_processes = min(num_processes or os.cpu_count(), os.cpu_count())

    def get_video_file_list(self) -> list[str]:
        return self.preprocess_files_fn(self.videos_dir)[: self.rows * self.cols]

    def load_videos(self, video_files: list[str]) -> list[np.ndarray]:
        h, w = self.input_resolution
        out = []
        for video_file in tqdm(video_files[:self.rows * self.cols]):
            out.append(self.load_video_fn(video_file, **{
                "video_dir": self.videos_dir,
                "height": h,
                "width": w,
                "fps": self.fps,
            }))
        return out
        

    def generate_grid(self, videos: list[np.ndarray]) -> list[np.ndarray]:
        frames = []
        margin_h, margin_w = self.margin_h // 2, self.margin_w // 2
        for t in trange(self.num_frames, desc="Generating Grids"):
            rows = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    cur_video = videos[i * self.cols + j]
                    _, cur_video_gen = cur_video
                    # cur_frame = cur_video[t % len(cur_video)]  # looping
                    cur_frame = next(cur_video_gen).astype(np.uint8)
                    # resize to 960x540
                    cur_frame = cv2.resize(cur_frame, (960, 540), interpolation=cv2.INTER_AREA)
                    video = cur_frame
                    h, w, c = video.shape
                    video_wrapped = (
                        np.ones((h + margin_h, w + margin_w, c), dtype=np.uint8) * 255
                    )
                    video_wrapped[
                        margin_h // 2 : margin_h // 2 + h,
                        margin_w // 2 : margin_w // 2 + w,
                    ] = video
                    row.append(video_wrapped)
                rows.append(np.concatenate(row, axis=1))
            frame = np.concatenate(rows, axis=0)
            frames.append(frame)
            break
        return frames

    def _center_video_frames(self, videos: list[np.ndarray]) -> int:
        """video will start zoom-out once the center videos is finished playing 1st time"""
        center_video_idx = (self.rows * self.cols - 1) // 2
        center_video = videos[center_video_idx]
        return len(center_video)

    def generate_final_frames(
        self, videos: list[np.ndarray], frames_grid: list[np.ndarray]
    ) -> list[np.ndarray]:
        H, W = self.final_resolution
        h1, w1, _ = frames_grid[0].shape
        ch1 = h1 // 2  # center h1
        cw1 = w1 // 2
        ph1 = h1 // self.rows  # patch h1
        pw1 = w1 // self.cols 
        # zoom_out_start = self._center_video_frames(videos)
        zoom_out_start = 0 * self.fps
        zoom_out_end = self.zoom_out_end_sec * self.fps
        margin_h, margin_w = self.margin_h // 2, self.margin_w // 2
        frames = []
        f_offset = 0
        for f in trange(self.num_frames, desc="generating final frames"):
            # if f <= zoom_out_start:
            #     oh1 = ph1 // 2 * self.num_start_rows - self.margin_h // 4
            #     ow1 = pw1 // 2 * self.num_start_cols - self.margin_w // 4
            #     h_min = ch1 - oh1
            #     h_max = ch1 + oh1
            #     w_min = cw1 - ow1
            #     w_max = cw1 + ow1

            if zoom_out_start < f - f_offset <= zoom_out_end:
                oh1 = int(
                    (h1 // 2 - ph1 // 2 * self.num_start_rows + self.margin_h // 4)
                    / (zoom_out_end - zoom_out_start)
                    * (f - f_offset - zoom_out_start)
                    + ph1 // 2 * self.num_start_rows
                    - self.margin_h // 4
                )
                ow1 = int(
                    (w1 // 2 - pw1 // 2 * self.num_start_cols + self.margin_w // 4)
                    / (zoom_out_end - zoom_out_start)
                    * (f - f_offset - zoom_out_start)
                    + pw1 // 2 * self.num_start_cols
                    - self.margin_w // 4
                )

                if (
                    self.mid_pause
                    and (
                        oh1 >= ph1 // 2 * self.r_mid - self.margin_h // 4
                        or ow1 >= pw1 // 2 * self.c_mid - self.margin_w // 4
                    )
                    and f_offset < zoom_out_start
                ):
                    f_offset += 1
                    oh1 = ph1 // 2 * self.r_mid - self.margin_h // 4
                    ow1 = pw1 // 2 * self.c_mid - self.margin_w // 4

                h_min = ch1 - oh1
                h_max = ch1 + oh1
                w_min = cw1 - ow1
                w_max = cw1 + ow1
            else:
                oh2 = h1 // 2
                ow2 = w1 // 2
                h_min = ch1 - oh2
                h_max = ch1 + oh2
                w_min = cw1 - ow2
                w_max = cw1 + ow2
            rows = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    cur_video = videos[i * self.cols + j]
                    video_file, cur_video_gen = cur_video
                    # cur_frame = cur_video[t % len(cur_video)]  # looping
                    # cur_frame = next(cur_video).astype(np.uint8)
                    try:
                        cur_frame = next(cur_video_gen).astype(np.uint8) # skip 1 frame
                    except:
                        videos[i*self.cols + j] = (video_file, skvideo.io.vreader(video_file))
                        cur_video_gen = videos[i*self.cols + j][1]
                        cur_frame = next(cur_video_gen).astype(np.uint8)

                    # resize to 960x540
                    cur_frame = cv2.resize(cur_frame, (960, 540), interpolation=cv2.INTER_AREA)
                    video = cur_frame
                    h, w, c = video.shape
                    video_wrapped = (
                        np.ones((h + margin_h, w + margin_w, c), dtype=np.uint8) * 255
                    )
                    video_wrapped[
                        margin_h // 2 : margin_h // 2 + h,
                        margin_w // 2 : margin_w // 2 + w,
                    ] = video
                    row.append(video_wrapped)
                rows.append(np.concatenate(row, axis=1))
            frame = np.concatenate(rows, axis=0).astype(np.uint8)
            frame = cv2.resize(
                frame[h_min:h_max, w_min:w_max],
                (W, H),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(frame)
        return frames

    def save_gallery(self, frames: list[np.ndarray]):
        os.makedirs(self.output_dir, exist_ok=True)
        raw_path = os.path.join(self.output_dir, f"{self.output_name}_raw.mp4")
        imageio.mimsave(raw_path, frames, fps=self.fps)
        self.compress(
            in_mp4_path=raw_path,
            out_mp4_path=os.path.join(self.output_dir, f"{self.output_name}.mp4"),
        )

    @staticmethod
    def compress(in_mp4_path: str, out_mp4_path: str, delete_input: bool = True):
        commands_list = [
            "ffmpeg",
            "-v",
            "quiet",
            "-y",
            "-i",
            in_mp4_path,
            "-vcodec",
            "libx264",
            "-crf",
            "28",
            out_mp4_path,
        ]
        assert subprocess.run(commands_list).returncode == 0, commands_list

    def process(self):
        """overall pipeline. can run separately outside for debugging"""
        video_files = self.get_video_file_list()
        videos = self.load_videos(video_files)
        frames_grid = self.generate_grid(videos)
        videos = self.load_videos(video_files)
        gallery = self.generate_final_frames(videos, frames_grid)
        self.save_gallery(gallery)


def reorder_files(videos_dir: str) -> list[str]:
    # with checker floor
    print(videos_dir)
    RNG = np.random.default_rng(seed=1)
    video_extra = glob.glob(videos_dir + "*.mp4")
    videos = glob.glob(videos_dir + "/*.mp4")
    RNG.shuffle(videos)
    # videos[44] = video_extra[0]
    # videos[45] = '/home/mdalal/research/optimus/release_video/pick_place/17_animation.mp4'
    # videos[54] = '/home/mdalal/research/optimus/release_video/microwave/17_animation.mp4'
    # videos[55] = '/home/mdalal/research/optimus/release_video/shelf/2_animation.mp4'
    return videos


def load_video(
    video_file: str, *, video_dir: str, height: int, width: int, fps: int
) -> np.ndarray:
    # cmd = (
    #     ffmpeg.input(os.path.join(video_dir, video_file))
    #     # .filter("fps", fps=fps)
    #     # .filter("scale", width, height)
    # )
    # out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
    #     capture_stdout=True, quiet=True
    # )
    # video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    # video = video[::2]
    # return video

    import skvideo.io  
    # videodata = skvideo.io.vread(os.path.join(video_dir, video_file), height=height, width=width, num_frames=480) #adding height, width causes blank frames for some reason?
    # assert videodata.shape[0] >= 480
    videodata = skvideo.io.vreader(os.path.join(video_dir, video_file))
    # subsample every other frame
    # videodata = videodata[::2, :, :, :]
    return os.path.join(video_dir, video_file), videodata
    


ZoomOut(
    videos_dir="/Users/murtaza/Documents/CMU/Research/MPRL-2023/planseqlearn.github.io/resources/clean",
    preprocess_files_fn=reorder_files,
    load_video_fn=load_video,
    output_dir="/Users/murtaza/Documents/CMU/Research/MPRL-2023/planseqlearn.github.io/resources",
    output_name='gallery',
).process()

