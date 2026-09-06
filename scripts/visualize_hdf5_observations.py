import argparse
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt


RGB_KEYS = {
    "Left RGB": "left_robot_camera_rgb",
    "Right RGB": "right_robot_camera_rgb",
    "Overhead RGB": "overhead_camera_rgb",
}

DEPTH_KEYS = {
    "Left Depth": "left_robot_camera_depth",
    "Right Depth": "right_robot_camera_depth",
    "Overhead Depth": "overhead_camera_depth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RGB and depth observations from a ScaleBench HDF5 dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output video path (.mp4 or .gif).",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode name. Defaults to the first episode.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS. Default: 10.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive playback window.",
    )
    return parser.parse_args()


def resolve_episode(data_group, requested_episode):
    if len(data_group) == 0:
        raise ValueError("Input HDF5 file contains no episodes.")

    episode_name = requested_episode or next(iter(data_group.keys()))

    if episode_name not in data_group:
        available = ", ".join(data_group.keys())
        raise KeyError(
            f"Episode '{episode_name}' not found. "
            f"Available episodes: {available}"
        )

    return episode_name


def validate_observation_keys(obs):
    required_keys = list(RGB_KEYS.values()) + list(DEPTH_KEYS.values())
    missing = [key for key in required_keys if key not in obs]

    if missing:
        raise KeyError(f"Missing observation keys: {missing}")


def create_animation(obs, episode_name, fps):
    frame_count = obs["left_robot_camera_rgb"].shape[0]

    if frame_count == 0:
        raise ValueError(f"Episode '{episode_name}' contains no frames.")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    rgb_artists = []
    depth_artists = []

    for ax, (title, key) in zip(axes[0], RGB_KEYS.items()):
        image = obs[key][0]
        artist = ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
        rgb_artists.append((artist, key))

    for ax, (title, key) in zip(axes[1], DEPTH_KEYS.items()):
        depth = obs[key][0].squeeze()
        artist = ax.imshow(depth)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(artist, ax=ax, fraction=0.046)
        depth_artists.append((artist, key))

    title = fig.suptitle(f"{episode_name} — frame 0/{frame_count - 1}")
    plt.tight_layout()

    def update(frame_idx):
        for artist, key in rgb_artists:
            artist.set_data(obs[key][frame_idx])

        for artist, key in depth_artists:
            depth = obs[key][frame_idx].squeeze()
            artist.set_data(depth)
            artist.set_clim(depth.min(), depth.max())

        title.set_text(
            f"{episode_name} — frame {frame_idx}/{frame_count - 1}"
        )

        return [
            title,
            *(artist for artist, _ in rgb_artists),
            *(artist for artist, _ in depth_artists),
        ]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )

    return fig, ani, frame_count


def save_animation(ani, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "Saving MP4 requires ffmpeg. "
                "Install ffmpeg or use a .gif output instead."
            )
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError(
            "Unsupported output format. Use .mp4 or .gif."
        )

    print(f"Saving animation to: {output_path}")
    ani.save(output_path, writer=writer)
    print("Save complete.")


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {args.input}")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0.")

    with h5py.File(args.input, "r") as f:
        if "data" not in f:
            raise KeyError("Input HDF5 file does not contain a 'data' group.")

        data_group = f["data"]
        episode_name = resolve_episode(data_group, args.episode)

        episode = data_group[episode_name]

        if "obs" not in episode:
            raise KeyError(
                f"Episode '{episode_name}' does not contain an 'obs' group."
            )

        obs = episode["obs"]
        validate_observation_keys(obs)

        fig, ani, frame_count = create_animation(
            obs=obs,
            episode_name=episode_name,
            fps=args.fps,
        )

        print(f"Input: {args.input}")
        print(f"Episode: {episode_name}")
        print(f"Frames: {frame_count}")
        print(f"FPS: {args.fps}")

        if args.output is not None:
            save_animation(ani, args.output, args.fps)

        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
