#!/usr/bin/env python3
import argparse

from radon_bridge.runtime.next_update_replay import compare_attempt


parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
parser.add_argument("--packet", required=True)
parser.add_argument("--gpu-job-id", required=True)
args = parser.parse_args()
print(compare_attempt(args.attempt, args.packet, gpu_job_id=args.gpu_job_id))

