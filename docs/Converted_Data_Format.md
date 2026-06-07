# Converted Data Format

```Python
# each mocap clip
clip = {
    '_id': { '$oid': String },
    'recordDate': Int,
    'animationClip': {
            'name': String,
            'duration': Decimal,
            'uuid': String
            'blendMode': Int,
            'tracks': [ Tracks ]
        }
    }
}
```

```Python
track = {
    'name': String, 
    'type': String, 
    'times': [ Decimals ], 
    'values': [ Decimals ], 
}
```

- For face (blendshapes) -> len(values) = len(times)       # blend value
- For normal body joints -> len(values) = 4 * len(times)   # quaternion

All quaternion joints:

```raw
'rootx.quaternion',
'spine_01x.quaternion',
'spine_02x.quaternion',
'neckx.quaternion',
'headx.quaternion',
'shoulderr.quaternion',
'arm_stretchr.quaternion',
'forearm_stretchr.quaternion',
'handr.quaternion',
'thumb1r.quaternion',
'thumb2r.quaternion',
'index1r.quaternion',
'index2r.quaternion',
'index3r.quaternion',
'middle1r.quaternion',
'middle2r.quaternion',
'middle3r.quaternion',
'ring1r.quaternion',
'ring2r.quaternion',
'ring3r.quaternion',
'pinky1r.quaternion',
'pinky2r.quaternion',
'pinky3r.quaternion',
'shoulderl.quaternion',
'arm_stretchl.quaternion',
'forearm_stretchl.quaternion',
'handl.quaternion',
'thumb1l.quaternion',
'thumb2l.quaternion',
'index1l.quaternion',
'index2l.quaternion',
'index3l.quaternion',
'middle1l.quaternion',
'middle2l.quaternion',
'middle3l.quaternion',
'ring1l.quaternion',
'ring2l.quaternion',
'ring3l.quaternion',
'pinky1l.quaternion',
'pinky2l.quaternion',
'pinky3l.quaternion',
'thigh_stretchl.quaternion',
'leg_stretchl.quaternion',
'footl.quaternion',
'thigh_stretchr.quaternion',
'leg_stretchr.quaternion',
'footr.quaternion'
```