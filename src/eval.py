import numpy as np


def eval_compression(raw_quats, compressed_quats, compress_info):
    per_joint_info = {}
    summary_info = dict(
        total_raw_quats = 0,
        total_raw_floats = 0,
        total_control_pts = 0,
        total_knots = 0,
        total_data_times = 0,
        total_compressed_floats = 0
    )
    
    joint_names = compressed_quats.keys()
    for joint in joint_names:
        raw_data = raw_quats[joint]
        com_data = compressed_quats[joint]
        compression_type = com_data["compression_type"]

        per_joint_info[joint] = {}
        joint_info = per_joint_info[joint]

        joint_info["total_raw_quats"]     = raw_data.shape[0]
        joint_info["total_raw_floats"]    = 4 * joint_info["total_raw_quats"]
        summary_info["total_raw_quats"]  += joint_info["total_raw_quats"]
        summary_info["total_raw_floats"] += joint_info["total_raw_floats"]
        
        if compression_type == "bspline":
            joint_info["total_control_pts"] = com_data["control_pts"].shape[0]
            joint_info["total_knots"]       = com_data["knot_vector"].shape[0]
            joint_info["total_data_times"]  = com_data["data_time"].shape[0]
        
        elif compression_type == "singular":
            joint_info["total_control_pts"] = 1
            joint_info["total_knots"]       = 0
            joint_info["total_data_times"]  = 1     # this is int, not float (but count as float for simplicity)
            
        joint_info["total_compressed_floats"] = 4 * joint_info["total_control_pts"] + joint_info["total_knots"] + joint_info["total_data_times"]
        joint_info["num_iterations"] = len(compress_info["error_history"][joint])
        # joint_info["final_error"] = compress_info["error_history"][joint][-1]
        
        summary_info["total_control_pts"] += joint_info["total_control_pts"]
        summary_info["total_knots"]       += joint_info["total_knots"]
        summary_info["total_data_times"]  += joint_info["total_data_times"]
        summary_info["total_compressed_floats"] += joint_info["total_compressed_floats"]

    return summary_info, per_joint_info
