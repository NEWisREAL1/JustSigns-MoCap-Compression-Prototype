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
        'blendshape_tracks_data': {
            'type': str,                    # type name for reconstruction (usaully "number" but stored just in case the naming change)
            'q_bits': int,                  # number of bits used in quantization of "values"
            'f_fps': int,                   # fps used in frame index encoding of "times",
            'groups': [                     # the list of unique groups of tracks
                {
                    'names': list[str],             # the blendshape names sharing the "times" and "values"
                    'f_times': list[int] -> str,    # frame index codes of "times" -> encoded as Base64
                    'q_values': {                   # the structure containing quantized "values"
                        'codes': list[int] -> str   # list of (unsigned) quantize codes -> encoded as Base64
                        'scale': float              # the scale factor for dequantizing
                        'zero' : float              # the zero point for dequantizing
                    }
                }
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

Arrange in the same way as raw format, but "times" and "values" of every tracks is Base64 encoded.

### 2. Direct Quantization

Arrage similar to raw format, but "times" and "values" are dict of "codes", "scale", and "zero":

```Python
{
    '_id': {'$oid': str},
    'animationClip': {
        'name': str,
        'duration': float,
        'tracks': [
            {
                'name'  : str,
                'q_times' : {
                    'codes': list[int] -> str   # list of (unsigned) quantize codes -> encoded as Base64
                    'scale': float              # the scale factor for dequantizing
                    'zero' : float              # the zero point for dequantizing
                },
                'q_values': {
                    'codes': list[int] -> str   # list of (unsigned) quantize codes -> encoded as Base64
                    'scale': float              # the scale factor for dequantizing
                    'zero' : float              # the zero point for dequantizing
                },
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