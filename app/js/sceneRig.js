import * as THREE from 'three';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

/**
 * One "sprite": a single positionable slot in the scene that holds both a
 * meshed clone and a skeleton-only clone of the same rig, plus a text label
 * above it. `mode` controls which of mesh/skeleton is visible -- 'mesh' or
 * 'skeleton' shows just one, 'both' shows them overlaid at the same spot,
 * instead of having two separately positioned rows like the viewer used to.
 */

const DEFAULT_LABEL = {
    text: '',
    fontSize: 22,
    paddingX: 14,
    paddingY: 8,
    borderRadius: 10,
    background: 'rgba(0,0,0,0.65)',
    color: '#00aaff',
    canvasWidth: 512,
    canvasHeight: 96,
    scaleX: 1.2,
    scaleY: 0.225,
    height: 2.0,
};

function buildLabelSprite(label) {
    const canvas = document.createElement('canvas');
    canvas.width = label.canvasWidth;
    canvas.height = label.canvasHeight;
    const ctx = canvas.getContext('2d');

    ctx.font = `bold ${label.fontSize}px sans-serif`;
    const textWidth = ctx.measureText(label.text).width;

    const bgWidth = Math.min(canvas.width, textWidth + label.paddingX * 2);
    const bgHeight = Math.min(canvas.height, label.fontSize + label.paddingY * 2);
    const bgX = (canvas.width - bgWidth) / 2;
    const bgY = (canvas.height - bgHeight) / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = label.background;
    ctx.beginPath();
    ctx.roundRect(bgX, bgY, bgWidth, bgHeight, label.borderRadius);
    ctx.fill();

    ctx.fillStyle = label.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label.text, canvas.width / 2, canvas.height / 2);

    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
    sprite.scale.set(label.scaleX, label.scaleY, 1);
    sprite.position.set(0, label.height, 0);
    return { sprite, height: label.height };
}

/**
 * @param {THREE.Object3D} templateScene - the loaded GLTF scene to clone from
 * @param {THREE.AnimationClip} clip
 * @param {object} options
 * @param {{x?:number,y?:number,z?:number}} [options.position]
 * @param {'mesh'|'skeleton'|'both'} [options.mode]
 * @param {boolean} [options.visible]
 * @param {number} [options.speed] - AnimationMixer.timeScale; 1 = normal speed
 * @param {string} [options.name]
 * @param {object} [options.label] - overrides merged onto DEFAULT_LABEL
 */
export function createRig(templateScene, clip, options = {}) {
    const group = new THREE.Group();

    const meshInstance = SkeletonUtils.clone(templateScene);
    group.add(meshInstance);

    const skeletonInstance = SkeletonUtils.clone(templateScene);
    skeletonInstance.traverse(node => {
        if (node.isMesh) node.visible = false;
    });
    group.add(skeletonInstance);

    // SkeletonHelper aliases its own `matrix` to root.matrixWorld and disables
    // matrixAutoUpdate -- it already *is* the root's full world transform, so
    // it must be added directly to the top-level scene. Nesting it under
    // skeletonInstance/group (both of which carry that same world transform)
    // multiplies the offset in twice, which is why the skeleton used to
    // render at 2x the rig's position.
    const helper = new THREE.SkeletonHelper(skeletonInstance);
    helper.material.linewidth = 3;

    const mixers = [
        new THREE.AnimationMixer(meshInstance),
        new THREE.AnimationMixer(skeletonInstance),
    ];
    mixers.forEach(mixer => mixer.clipAction(clip).play());

    const label = { ...DEFAULT_LABEL, ...options.label, text: options.name ?? DEFAULT_LABEL.text };
    const { sprite: labelSprite } = buildLabelSprite(label);
    group.add(labelSprite);

    const state = {
        visible: options.visible ?? true,
        mode: options.mode ?? 'mesh',
        position: { x: 0, y: 0, z: 0, ...options.position },
        speed: options.speed ?? 1,
    };

    function applySpeed() {
        // AnimationMixer.timeScale multiplies the delta each mixer advances
        // by, so this scales playback without touching the clip's own
        // times/tracks -- slowing down is just timeScale < 1.
        mixers.forEach(mixer => { mixer.timeScale = state.speed; });
    }

    function applyVisibility() {
        meshInstance.visible = state.visible && (state.mode === 'mesh' || state.mode === 'both');
        skeletonInstance.visible = state.visible && (state.mode === 'skeleton' || state.mode === 'both');
        helper.visible = skeletonInstance.visible;
        labelSprite.visible = state.visible;
    }

    function applyPosition() {
        group.position.set(state.position.x, state.position.y, state.position.z);
    }

    applyVisibility();
    applyPosition();
    applySpeed();

    return {
        name: options.name ?? '',
        group,
        helper, // must be added to the top-level scene, NOT to `group` -- see note above

        get visible() { return state.visible; },
        set visible(value) { state.visible = value; applyVisibility(); },

        get mode() { return state.mode; },
        set mode(value) { state.mode = value; applyVisibility(); },

        get position() { return { ...state.position }; },
        setPosition(partial) {
            Object.assign(state.position, partial);
            applyPosition();
        },

        get speed() { return state.speed; },
        set speed(value) { state.speed = value; applySpeed(); },

        update(delta) {
            mixers.forEach(mixer => mixer.update(delta));
        },
    };
}
