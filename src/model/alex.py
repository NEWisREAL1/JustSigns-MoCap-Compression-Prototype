from pygltflib import GLTF2

from src.model.skeleton import KinematicsSkeleton


def get_alex_skeleton(clean_names=True, includes_face=False):
    gltf = GLTF2.load("model/Alex_Rig_v2.4_rokoko_wface_nov30.glb")
    nodes = gltf.nodes

    queue = [140] # alex_idex of root.x
    count = 1
    idx_mapper = { 140: 0 } # map alex model index to ordered

    names = [nodes[140].name]
    if clean_names:
        names[0] = names[0].replace(".", "")
    parents = [-1]
    translations = [nodes[140].translation]
    rotations = [nodes[140].rotation]

    while queue:
        parent_idx = queue.pop(0)
        node = nodes[parent_idx]

        if not includes_face and parent_idx == 73:
            # not adding children of "head.x" (alex_idx=73), to exclude faces
            continue

        for child_idx in node.children[::-1]:
            names.append(nodes[child_idx].name)
            if clean_names:
                names[-1] = names[-1].replace(".", "")
            parents.append(idx_mapper[parent_idx])
            translations.append(nodes[child_idx].translation)
            rotations.append(nodes[child_idx].rotation)

            queue.append(child_idx)
            idx_mapper[child_idx] = count
            count += 1

    return KinematicsSkeleton(names, parents, translations, rotations)