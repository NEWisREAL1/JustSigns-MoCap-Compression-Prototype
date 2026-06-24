import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { compileAnimationClip } from './codec.js';
import { createRig } from './sceneRig.js';
import { buildControlPanel } from './controlPanel.js';

const CONFIG = {
    modelUrl: '../model/Alex_Rig_v2.4_rokoko_wface_nov30.glb',
    spacingX: 1.0, // fallback spacing for clips that don't set an explicit position.x
    clips: [
        // Per-clip overrides: `position: { x, y, z }`, `mode: 'mesh' | 'skeleton'`,
        // `visible: false`, `label: { fontSize, background, color, ... }`.
        { name: '481 Original'         , url: '../data/json/clip_481.json' },
        { name: '481 RawBase64'        , url: '../out/rb64/clip_481_rb64.json' },
        { name: '481 Quantized'        , url: '../out/quan/clip_481_quan.json' },
        { name: '481 BlendshapesScheme', url: '../out/bsch/clip_481_bsch.json' },
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

const clock = new THREE.Clock();
const rigs = [];

const loadingElement = document.getElementById('loading');
const errorLogElement = document.getElementById('error-log');
const controlPanelElement = document.getElementById('rig-panel');

async function loadClipFile(info) {
    try {
        const response = await fetch(info.url);
        if (!response.ok) throw new Error(response.statusText || String(response.status));
        return { config: info, data: await response.json() };
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
            const { config } = clipObj;
            const defaultX = (index - (activeClips.length - 1) / 2) * CONFIG.spacingX;
            const position = { x: defaultX, y: 0, z: 0, ...config.position };

            const threeClip = compileAnimationClip(clipObj.data);
            const rig = createRig(gltf.scene, threeClip, {
                name: config.name,
                position,
                mode: config.mode,
                visible: config.visible,
                label: config.label,
            });

            scene.add(rig.group);
            scene.add(rig.helper);
            rigs.push(rig);
        });

        if (controlPanelElement) {
            buildControlPanel(controlPanelElement, rigs);
        }

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
    rigs.forEach(rig => rig.update(delta));
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

init();
