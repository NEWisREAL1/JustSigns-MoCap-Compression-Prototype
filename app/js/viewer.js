import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

import { compileAnimationClip } from './codec.js';

const CONFIG = {
    modelUrl: '../model/Alex_Rig_v2.4_rokoko_wface_nov30.glb',
    spacingX: 1.5,
    spacingZ: 2.0,
    clips: [
        { name: 'Original', url: '../data/json/clip_73.json' },
        { name: 'RawBase64', url: '../out/raw_base64/clip_rb64_73.json' },
        { name: 'QuantizedB64', url: '../out/quantize/clip_quan_73.json' }
    ]
};

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x1a1a1a, 10, 100);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);
scene.add(new THREE.GridHelper(50, 50, 0x888888, 0x333333));

const meshGroup = new THREE.Group();
const skeletonGroup = new THREE.Group();
skeletonGroup.visible = false
scene.add(meshGroup);
scene.add(skeletonGroup);

const clock = new THREE.Clock();
const mixers = [];

const loadingElement = document.getElementById('loading');
const errorLogElement = document.getElementById('error-log');
const meshToggle = document.getElementById('toggle-meshes');
const skeletonToggle = document.getElementById('toggle-skeletons');

function attachAnimation(model, clip) {
    const mixer = new THREE.AnimationMixer(model);
    mixer.clipAction(clip).play();
    mixers.push(mixer);
}

async function loadClipFile(info) {
    try {
        const response = await fetch(info.url);
        if (!response.ok) throw new Error(response.statusText || String(response.status));
        return { name: info.name, data: await response.json() };
    } catch {
        if (errorLogElement) {
            errorLogElement.innerHTML += `Failed: ${info.url}<br/>`;
        }
        return null;
    }
}

async function init() {
    try {
        const gltf = await new Promise((resolve, reject) => new GLTFLoader().load(CONFIG.modelUrl, resolve, undefined, reject));

        const loadedClips = await Promise.all(CONFIG.clips.map(loadClipFile));
        const activeClips = loadedClips.filter(Boolean);

        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        activeClips.forEach((clipObj, index) => {
            const xPos = (index - (activeClips.length - 1) / 2) * CONFIG.spacingX;
            const threeClip = compileAnimationClip(clipObj.data);

            const meshClone = SkeletonUtils.clone(gltf.scene);
            meshClone.position.set(xPos, 0, 0);
            meshGroup.add(meshClone);
            attachAnimation(meshClone, threeClip);

            const skelClone = SkeletonUtils.clone(gltf.scene);
            skelClone.traverse(node => {
                if (node.isMesh) node.visible = false;
            });
            const helper = new THREE.SkeletonHelper(skelClone);
            helper.material.linewidth = 3;
            skelClone.add(helper);
            skelClone.position.set(xPos / 2, 0, -CONFIG.spacingZ);
            skeletonGroup.add(skelClone);
            attachAnimation(skelClone, threeClip);

            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.roundRect(0, 0, 512, 128, 20);
            ctx.fill();
            ctx.font = 'bold 36px sans-serif';
            ctx.fillStyle = '#00aaff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(clipObj.name, 256, 64);

            const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas) }));
            sprite.position.set(xPos, 2.3, 0);
            sprite.scale.set(1.5, 0.375, 1);
            scene.add(sprite);
        });

        camera.position.set(0, 2, activeClips.length * 1.5 + 4);
        controls.update();
        animate();
    } catch (error) {
        console.error(error);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const delta = clock.getDelta();
    mixers.forEach(mixer => mixer.update(delta));
    renderer.render(scene, camera);
}

meshToggle?.addEventListener('change', event => {
    meshGroup.visible = event.target.checked;
});

skeletonToggle?.addEventListener('change', event => {
    skeletonGroup.visible = event.target.checked;
});

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

init();