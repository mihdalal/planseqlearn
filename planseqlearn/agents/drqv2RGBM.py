import numpy as np

from planseqlearn.agents.drqv2 import DrQV2Agent


class DrQV2RGBMAgent(DrQV2Agent):
    def get_frames_to_record(self, obs):
        rgbm_obs = obs["pixels"]
        frame = rgbm_obs[-4:].transpose(1, 2, 0).clip(0, 255)
        rgb_frame = frame[:, :, :3]
        mask_frame = frame[:, :, 3:]
        return {"rgb": rgb_frame.astype(np.uint8), "mask": mask_frame.astype(np.uint8)}
