import argparse
import os
import sys

from src._compression import ClipCompressor
from src.compressors.bspline import FixedSizeBSplineCompressor
from src.compressors.colinear import ColinearDecimatoinCompressor
from src.compressors.quantization import QuantizeCompressor
from src.compressors.raw_base64 import RawBase64Compressor

from src.cagd.lspia import LSPIASolver
from src.data import (
    extract_tracks,
    load_clip,
    pack_anim_to_array,
    save_clip,
)
from src.eval import mean_per_joint_position_error
from src.model.alex import get_alex_skeleton
from src.processing import AntipodalPreprocessor


def main(clip_idx):
    print(">>> Initiating ...")
    original_path = f"data/json/clip_{clip_idx}.json"
    raw_base64_path = f"out/raw_base64/clip_rb64_{clip_idx}.json"

    compressors = dict(
        Quantize=dict(
            com=ClipCompressor(
                number_compressor=QuantizeCompressor(),
                quanternion_compressor=QuantizeCompressor(),
            ),
            path=f"out/quantize/clip_quan_{clip_idx}.json"
        ),
        BSpline_CLD20=dict(
            com=ClipCompressor(
                number_compressor=ColinearDecimatoinCompressor(),
                quanternion_compressor=FixedSizeBSplineCompressor(LSPIASolver(initial_num_cps=20)),
                preprocessors=[AntipodalPreprocessor()],
            ),
            path=f"out/temp/clip_bspl_20_{clip_idx}.json",
        ),
        BSpline_CLD40=dict(
            com=ClipCompressor(
                number_compressor=ColinearDecimatoinCompressor(),
                quanternion_compressor=FixedSizeBSplineCompressor(LSPIASolver(initial_num_cps=40)),
                preprocessors=[AntipodalPreprocessor()],
            ),
            path=f"out/temp/clip_bspl_40_{clip_idx}.json",
        ),
        BSpline_CLD60=dict(
            com=ClipCompressor(
                number_compressor=ColinearDecimatoinCompressor(),
                quanternion_compressor=FixedSizeBSplineCompressor(LSPIASolver(initial_num_cps=60)),
                preprocessors=[AntipodalPreprocessor()],
            ),
            path=f"out/temp/clip_bspl_60_{clip_idx}.json",
        ),
    )

    all_paths = [raw_base64_path] + [c["path"] for c in compressors.values()]
    for path in all_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # ----- Load Original Clip ----- #

    original_clip = load_clip(original_path)
    # print_json_structure(original_clip)
    print("loaded original clip")

    if not extract_tracks(original_clip["animationClip"]["tracks"], "number"):
        print("note: this clip does NOT contain blendshapes")

    # ----- Baseline Encodings ----- #

    rb64_compress = ClipCompressor(
        number_compressor=RawBase64Compressor(),
        quanternion_compressor=RawBase64Compressor(),
    )

    raw_base64_clip = rb64_compress.compress(original_clip)
    save_clip(raw_base64_clip, raw_base64_path)
    # print_json_structure(raw_base64_clip)
    print(f"compressed - {"RawBase64":<30} -> outputed at {raw_base64_path}")

    # ----- Compressions ----- #

    compressed_clips = {}

    for key, data in compressors.items():
        com_clip = data["com"].compress(original_clip)
        compressed_clips[key] = com_clip
        save_clip(com_clip, data["path"])
        print(f"compressed - {key:<30} -> outputed at {data["path"]}")

    print(">>> Successfully Compress the Clip to All Format")
    print(">>> Starting Evaluation ...")

    # ----- Reconstruction & Evaluation ----- #
    
    alex = get_alex_skeleton()
    original_anim_arr = pack_anim_to_array(original_clip, alex)
    original_pos_arr, _ = alex.forward_kinematics(original_anim_arr)
    
    recon_raw_base64_clip = rb64_compress.decompress(raw_base64_clip)
    recon_raw_base64_anim_arr = pack_anim_to_array(recon_raw_base64_clip, alex)
    recon_raw_base64_pos_arr, _ = alex.forward_kinematics(recon_raw_base64_anim_arr)
    raw_base64_mpjpe = mean_per_joint_position_error(original_pos_arr, recon_raw_base64_pos_arr)

    original_disk_size = os.path.getsize(original_path) / 1024
    raw_base64_disk_size = os.path.getsize(raw_base64_path) / 1024
    save_raw_base64_original   = 100 - 100 * raw_base64_disk_size / original_disk_size
    
    all_mpjpe = {}
    all_sizes = {}

    for key, clip in compressed_clips.items():
        recon_clip = compressors[key]["com"].decompress(clip)
        recon_anim_arr = pack_anim_to_array(recon_clip, alex)
        recon_pos_arr, _ = alex.forward_kinematics(recon_anim_arr)
        mpjpe = mean_per_joint_position_error(original_pos_arr, recon_pos_arr)
        all_mpjpe[key] = mpjpe

        disk_size_KB = os.path.getsize(compressors[key]["path"]) / 1024
        save_vs_original   = 100 - 100 * disk_size_KB / original_disk_size
        save_vs_raw_base64 = 100 - 100 * disk_size_KB / raw_base64_disk_size
        all_sizes[key] = {}
        all_sizes[key]["KB"] = disk_size_KB
        all_sizes[key]["save_original"] = save_vs_original
        all_sizes[key]["save_raw_base64"] = save_vs_raw_base64

    # ----- Preservatoin Performance ----- #

    print("\n============= Error Evaluation =============")
    print(f"{"RawBase64":<20} MPJPE: {raw_base64_mpjpe:.10f}")
    
    for key, value in all_mpjpe.items():
        print(f"{key:<20} MPJPE: {value:.10f}")

    # ----- Compression Performance ----- #

    print("\n===== Compression Performace (On-Disk) =====")
    print(f"{"Original":<20} : {original_disk_size:,.0f} KB")
    print(f"{"RawBase64":<20} : {f"{raw_base64_disk_size:,.0f} KB":<10} — {save_raw_base64_original:,.2f}% saved from Original")

    for key, data in all_sizes.items():
        size_str = f"{data["KB"]:,.0f} KB"
        print(f"{key:<20} : {size_str:<10} — {data["save_original"]:,.2f}% saved from Original — {data["save_raw_base64"]:,.2f}% saved from RawBase64")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("clip_idx", type=int, help="Index of the clip to process, e.g. 19 for data/json/clip_19.json")
    args = parser.parse_args()

    main(args.clip_idx)
