import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a ScaleBench HDF5 dataset to LeRobot format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input HDF5 dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot repository ID, e.g. 'user/bottle_pick_place'.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name stored in the LeRobot dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Dataset frame rate. Default: 10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser.parse_args()


def get_features():
    return {
        "observation.images.left": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.overhead": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.depth.left": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
            "info": {"is_depth_map": True},
        },
        "observation.depth.right": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
            "info": {"is_depth_map": True},
        },
        "observation.depth.overhead": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
            "info": {"is_depth_map": True},
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action"],
        },
    }


def convert_episode(dataset, episode, task):
    frame_num = episode["actions"].shape[0]

    for i in range(frame_num):
        if i % 10 == 0:
            print(f"  frame {i}/{frame_num}")

        obs = episode["obs"]

        state = np.concatenate(
            [
                obs["left_arm_joint_pos"][i],
                obs["left_gripper_joint_pos"][i],
                obs["right_arm_joint_pos"][i],
                obs["right_gripper_joint_pos"][i],
            ]
        ).astype(np.float32)

        frame = {
            "observation.images.left": obs["left_robot_camera_rgb"][i],
            "observation.images.right": obs["right_robot_camera_rgb"][i],
            "observation.images.overhead": obs["overhead_camera_rgb"][i],
            "observation.depth.left": obs["left_robot_camera_depth"][i],
            "observation.depth.right": obs["right_robot_camera_depth"][i],
            "observation.depth.overhead": obs["overhead_camera_depth"][i],
            "observation.state": state,
            "action": episode["actions"][i].astype(np.float32),
            "task": task,
        }

        dataset.add_frame(frame)

    dataset.save_episode()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output}\n"
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=get_features(),
        root=args.output,
    )

    with h5py.File(args.input, "r") as f:
        if "data" not in f:
            raise KeyError("Input HDF5 file does not contain a 'data' group.")

        data_group = f["data"]

        for episode_name in data_group.keys():
            print(f"Processing episode: {episode_name}")
            convert_episode(
                dataset=dataset,
                episode=data_group[episode_name],
                task=args.task,
            )

    dataset.finalize()
    print(f"Conversion complete: {args.output}")


if __name__ == "__main__":
    main()