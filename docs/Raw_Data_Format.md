# Raw Data Format

```Python
# each mocap clip
clip = {
    '_id': { '$oid': String },
    'recordDate': Int,
    'animationRaw': {
        'version': String,
        'fps': Int,
        'scene': {
            # each member = mocap at a certain frame
            'timestamp': [ Decimals ],
            'actors': [ Actors ],
            'newtons': [ ... ],
            'prop': [ ... ],
            'characters': [ ... ]
        }
    }
}
```

```Python
actor = [{  # Extra outer array (why ?)
    'name': String,
    'color': [ IntTriplet ],
    'meta: {
        'hasGloves': Bool,
        'hasLeftGlove': Bool,
        'hasRightGlove': Bool,
        'hasBody': Bool,
        'hasFace': Bool
    },
    'dimensions': {
        'totalHeight': Decimal,
        'hipHeight': Decimal
    },
    'body': {
        'hip'   : {'position': {'x','y','z'}, 'Rotation': {'x','y','z','w'}},
        'spine' : {'position': {'x','y','z'}, 'Rotation': {'x','y','z','w'}},
        'chest' : {'position': {'x','y','z'}, 'Rotation': {'x','y','z','w'}},
        ...
    },
    'face': {
        'mouthFunnel'         : Decimal,
        'mouthLowerDownRight' : Decimal,
        'browDownLeft'        : Decimal,
        ...
    }
}]
```

All body parts

```raw
0   hip
1   spine
2   chest
3   neck
4   head
5   leftShoulder
6   leftUpperArm
7   leftLowerArm
8   leftHand
9   rightShoulder
10  rightUpperArm
11  rightLowerArm
12  rightHand
13  leftUpLeg
14  leftLeg
15  leftFoot
16  leftToe
17  leftToeEnd
18  rightUpLeg
19  rightLeg
20  rightFoot
21  rightToe
22  rightToeEnd
23  leftThumbProximal
24  leftThumbMedial
25  leftThumbDistal
26  leftThumbTip
27  leftIndexProximal
28  leftIndexMedial
29  leftIndexDistal
30  leftIndexTip
31  leftMiddleProximal
32  leftMiddleMedial
33  leftMiddleDistal
34  leftMiddleTip
35  leftRingProximal
36  leftRingMedial
37  leftRingDistal
38  leftRingTip
39  leftLittleProximal
40  leftLittleMedial
41  leftLittleDistal
42  leftLittleTip
43  rightThumbProximal
44  rightThumbMedial
45  rightThumbDistal
46  rightThumbTip
47  rightIndexProximal
48  rightIndexMedial
49  rightIndexDistal
50  rightIndexTip
51  rightMiddleProximal
52  rightMiddleMedial
53  rightMiddleDistal
54  rightMiddleTip
55  rightRingProximal
56  rightRingMedial
57  rightRingDistal
58  rightRingTip
59  rightLittleProximal
60  rightLittleMedial
61  rightLittleDistal
62  rightLittleTip
``` 

All face parts:

```raw
0   mouthFunnel
1   mouthLowerDownRight
2   browDownLeft
3   jawLeft
4   jawOpen
5   mouthDimpleRight
6   mouthDimpleLeft
7   eyeLookDownLeft
8   eyeLookDownRight
9   eyeLookInLeft
10  cheekSquintRight
11  mouthClose
12  mouthSmileLeft
13  mouthSmileRight
14  mouthPressRight
15  mouthLeft
16  mouthLowerDownLeft
17  mouthRight
18  mouthPressLeft
19  mouthRollUpper
20  mouthPucker
21  mouthRollLower
22  mouthShrugUpper
23  mouthShrugLower
24  mouthUpperUpRight
25  mouthStretchRight
26  mouthStretchLeft
27  browInnerUp
28  browOuterUpLeft
29  browDownRight
30  browOuterUpRight
31  cheekSquintLeft
32  eyeLookUpRight
33  cheekPuff
34  eyeLookOutLeft
35  eyeLookOutRight
36  eyeLookInRight
37  eyeWideLeft
38  jawForward
39  eyeWideRight
40  mouthFrownLeft
41  mouthFrownRight
42  eyeSquintRight
43  eyeLookUpLeft
44  mouthUpperUpLeft
45  eyeSquintLeft
46  jawRight
47  noseSneerLeft
48  noseSneerRight
49  tongueOut
50  eyeBlinkLeft
51  eyeBlinkRight
``` 