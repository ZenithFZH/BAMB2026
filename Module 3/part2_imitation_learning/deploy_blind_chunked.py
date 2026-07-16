"""Run a trained BlindChunkedClone on the SO-101 follower arm. No camera needed.

The policy predicts a chunk of future actions from the current joint angles;
we execute the first half of each chunk, then re-plan from the live state.

Safety: actions are clamped to the joint ranges seen in the training data, and
lerobot's max_relative_target rate-limits every motor step. Ctrl+C stops
cleanly and releases torque.

Usage:
    # First: full loop, prints actions, arm holds still
    uv run python deploy_blind_chunked.py --checkpoint blind_chunked.pt \
        --port /dev/tty.usbmodem5B610326911 --dry-run

    # Then: the real thing (hand near the power switch the first time)
    uv run python deploy_blind_chunked.py --checkpoint blind_chunked.pt \
        --port /dev/tty.usbmodem5B610326911 --seconds 30
"""

import argparse
import time

import torch
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from train_blind_chunked import BlindChunkedClone

FPS = 30
JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="bamb_follower")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=True)
    stats = {k: torch.zeros(ckpt["d_in"] if k.startswith("x") else ckpt["d_out"])
             for k in ("x_mean", "x_std", "y_mean", "y_std")}
    policy = BlindChunkedClone(ckpt["d_in"], ckpt["d_out"], ckpt["chunk_size"], stats)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    low, high = ckpt["action_low"], ckpt["action_high"]

    robot = SOFollower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, max_relative_target=10.0,
    ))
    robot.connect()
    print(f"connected ({'DRY RUN — no actions sent' if args.dry_run else 'LIVE'})")

    n_exec = ckpt["chunk_size"] // 2  # execute half the chunk, then re-plan
    t_end = time.perf_counter() + args.seconds
    try:
        while time.perf_counter() < t_end:
            obs = robot.get_observation()
            state = torch.tensor(
                [[obs[f"{j}.pos"] for j in JOINTS]], dtype=torch.float32
            )
            with torch.no_grad():
                chunk = policy(state)[0]  # (chunk_size, 6)
            chunk = torch.clamp(chunk, low, high)

            for step in range(n_exec):
                t_next = time.perf_counter() + 1 / FPS
                action = {
                    f"{j}.pos": chunk[step, i].item() for i, j in enumerate(JOINTS)
                }
                if args.dry_run:
                    if step == 0:
                        print("state ", [f"{v:7.1f}" for v in state[0].tolist()])
                        print("action", [f"{v:7.1f}" for v in chunk[0].tolist()])
                else:
                    robot.send_action(action)
                if time.perf_counter() > t_end:
                    break
                time.sleep(max(0.0, t_next - time.perf_counter()))
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        robot.disconnect()
        print("disconnected, torque released")


if __name__ == "__main__":
    main()
