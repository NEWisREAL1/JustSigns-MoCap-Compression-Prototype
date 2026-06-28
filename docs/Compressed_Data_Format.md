# Data Format for a Compressed MoCap Clip

This document provide the detail of the JSON structure used in compressed MoCap clips.

## Raw Clip Format

For reference, this is the structure of the original data used to feed to ThreeJS rendering engine:

```Python
{
    '_id': {'$oid': str},
    'animationClip': {
        'name': str,
        'duration': float,
        'tracks': [
            {
                'name'  : str,
                'times' : list[float],
                'values': list[float],
                'type'  : str,
            },
            ...
        ],
        'uuid': str,
        'blendMode': int
        },
    'recordDate': int
}
```

## A Compressed Clip

A compressed clip is structured as follows:

```Python
{
    '_id': {'$oid': str},
    'recordDate': int
    'animationClip': {
        'name': str,
        'duration': float,
        'uuid': str,
        'blendMode': int
        'blendshape_tracks_data': {global_data + tracks_data}, 
        'quaternion_tracks_data': {global_data + tracks_data}, 
    },
}
```

- `blendshape_tracks_data` and `quaternion_tracks_data` are dicts consisting of all neccessary data needed to reconstruct the render-able data, splited ito blenshapes and quaternion for different processing method on the two types.
- The two tracks data dicts should mandatorily contain keys `type_name` and `compress_type` that specify the original tracks type name (e.g., "number" or "quatetnion") and the type of compression (e.g., "raw_base64" or "blendshapes_scheme") repectively.

## Purposed Format

This is the format of the grand propesed scheme on MoCap compression:

### 1. The Blendshapes Scheme

```Python
'blendshape_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'q_bits': int,                  # number of bits used in quantization of "values"
    'f_fps': int,                   # fps used in frame index encoding of "times",
    'f_bits': int,                  # number of bits used in encoding of "times",
    'groups': [                     # the list of unique groups of tracks
        {
            'names': list[str],                 # the blendshape names sharing the "times" and "values"
            'f_times_b64': list[int] -> str,    # frame index codes of "times" -> encoded as Base64
            'q_values': {                       # the structure containing quantized "values"
                'codes_b64': list[int] -> str   # list of (unsigned) quantize codes -> encoded as Base64
                'scale': float                  # the scale factor for dequantizing
                'zero' : float                  # the zero point for dequantizing
            }
        },
        ...
    ]
}
```

### 2. The Quaternions Scheme

```Python
'quaternion_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'q_bits': int,                  # number of bits used in quantizations
    'bake_fps': int,                # fps used in baking provess, use this with the duration of the clip to determine the "times"
    'res_fps': int,            # fps used in capturing the residuals
    'tracks': [
        {
            'name': str,
            'control_points': {                 # dict containing quantized data of control points
                'x': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'y': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'z': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'w': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
            },
            'knots': {                          # dict containing quantized data of knot vector
                'codes_b64': list[int] -> str,  # should decode to (m,) or (pcs,)
                'scale': float,
                'zero': float,
            },
            'residuals': {                      # dict containing quantized data of residuals
                'x': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'y': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'z': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                'w': { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
            },
        },
        ...
    ]
    'eigen_tracks': [
        {
            'names': list[str],
            'num_pc': int,
            'eigenvecs': [      # list of (num_pcs) quantized eigenvecs, each of (num_joints) rows
                { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                ...
            ],
            'pc_scores': [      # list of (num_pcs) quuantized PC scores, each of (num_frame) rows
                { 'codes_b64': list[int] -> str, 'scale': list[float], 'zero': list[float] },
                ...
            ],
            'mean_b64': {
                'codes_b64': list[int] -> str,
                'scale': float,
                'zero': float,
            },    # list of (num_joints) entries
        }
    ],
    'static_tracks': [
        {
           'name': str,
           'value': list[float], 
        },
        ...
    ],
}
```

- `isEigen` specify whether the track represents the principal components (PCA track) of a group of tracks (`isEigen = True`) or raw data of a track (`isEigen = False`).
- If the track is a PCA track, `eigen_data` will contain the PCA data needed for reconstruction, otherwise `eigen_data` will not be part of the track.
- This scheme allow for decoding at varying resolution/fps (no "times" specify).
- The "times" for "residuals" can be reconstructed from `res_fps` and duration.

## Format for Baselines (Evaluation Only Format)

### 1. Base64 Encoding

The same set of "times" and "values" as in raw data, encoded as Base64:

```Python
'blendshape_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'tracks': [
        {
            'name': str,
            'times_b64' : list[float] -> str,   # "times" data -> encoded as Base64
            'values_b64': list[float] -> str,   # "values" data -> encoded as Base64
        },
        ...
    ]
}
```

```Python
'quaternion_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'tracks': [
        {
            'name': str,
            'times_b64' : list[float] -> str,   # "times" data -> encoded as Base64
            'values_b64': list[float] -> str,   # "values" data -> encoded as Base64
        },
        ...
    ]
}
```

### 2. Direct Quantization

Direct quantization on "times" and "values" of raw data:

```Python
'blendshape_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'q_bits': int,
    'tracks': [
        {
            'name': str,
            'q_times': {                                # dict containing quantization variables for "times"
                'codes_b64': list[int] -> str
                'scale': float
                'zero' : float
            },
            'q_values': {                               # dict containing quantization variables for "values"
                'codes_b64': list[int] -> str
                'scale': float
                'zero' : float
            },
        },
        ...
    ]
}
```

```Python
'quaternion_tracks_data': {
    'compress_type': str,
    'type_name': str,
    'q_bits': int,
    'tracks': [
        {
            'name': str,
            'q_times': {                                # dict containing quantization variables for "times"
                'codes_b64': list[int] -> str,
                'scale': float,
                'zero' : float,
            },
            'q_values': {                               # dict containing quantization variables for "values"
                'x' { 'codes_b64': list[int] -> str, 'scale': float, 'zero' : float },
                'y' { 'codes_b64': list[int] -> str, 'scale': float, 'zero' : float },
                'z' { 'codes_b64': list[int] -> str, 'scale': float, 'zero' : float },
                'w' { 'codes_b64': list[int] -> str, 'scale': float, 'zero' : float },
            },
        },
        ...
    ]
}
```