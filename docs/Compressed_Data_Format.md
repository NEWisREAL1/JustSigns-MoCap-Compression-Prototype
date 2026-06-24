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

```Python
{
    '_id': {'$oid': str},
    'recordDate': int
    'animationClip': {
        'name': str,
        'duration': float,
        'uuid': str,
        'blendMode': int
        'blendshape_tracks_data': {
            'compress_type': str,
            'type_name': str,
            'q_bits': int,                  # number of bits used in quantization of "values"
            'f_fps': int,                   # fps used in frame index encoding of "times",
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
        },
        'quaternion_tracks_data': {
            # TO BE DONE
        },
    },
}
```

## Format for Baselines (Evaluation Only Format)

### 1. Base64 Encoding

The same set of "times" and "values" as in raw data, encoded as Base64:

```Python
{
    '_id': {'$oid': str},
    'recordDate': int
    'animationClip': {
        'name': str,
        'duration': float,
        'uuid': str,
        'blendMode': int
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
        },
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
        },
    },
}
```

### 2. Direct Quantization

Direct quantization on "times" and "values" of raw data:

```Python
{
    '_id': {'$oid': str},
    'recordDate': int
    'animationClip': {
        'name': str,
        'duration': float,
        'uuid': str,
        'blendMode': int
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
        },
        'quaternion_tracks_data': {
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
        },
    },
}
```