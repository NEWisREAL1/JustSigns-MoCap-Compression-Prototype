/**
 * A minimal per-rig inspector: one row per loaded clip with a visibility
 * toggle, a mesh/skeleton mode switch, and X/Y/Z position fields. Pure DOM,
 * no framework -- rebuilt once after rigs are created via `buildControlPanel`.
 * Also includes one global playback-speed slider applied to every rig.
 */

function makeSpeedControl(rigs) {
    const wrapper = document.createElement('div');
    wrapper.className = 'rig-speed-control';

    const label = document.createElement('label');
    label.textContent = 'Speed';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '4';
    slider.step = '0.1';
    slider.value = '1';

    const readout = document.createElement('span');
    readout.className = 'rig-speed-readout';
    const renderReadout = value => { readout.textContent = `${Number(value).toFixed(2)}x`; };
    renderReadout(slider.value);

    slider.addEventListener('input', () => {
        renderReadout(slider.value);
        const speed = parseFloat(slider.value);
        rigs.forEach(rig => { rig.speed = speed; });
    });

    wrapper.append(label, slider, readout);
    return wrapper;
}

function makePositionField(axis, rig) {
    const wrapper = document.createElement('label');
    wrapper.className = 'rig-pos-field';
    wrapper.textContent = axis.toUpperCase();

    const input = document.createElement('input');
    input.type = 'number';
    input.step = '0.1';
    input.value = rig.position[axis];
    input.addEventListener('input', () => {
        const value = parseFloat(input.value);
        rig.setPosition({ [axis]: Number.isFinite(value) ? value : 0 });
    });

    wrapper.appendChild(input);
    return wrapper;
}

function makeRigRow(rig) {
    const row = document.createElement('div');
    row.className = 'rig-row';

    const header = document.createElement('div');
    header.className = 'rig-row-header';

    const visibleToggle = document.createElement('input');
    visibleToggle.type = 'checkbox';
    visibleToggle.title = 'Show / hide';
    visibleToggle.checked = rig.visible;
    visibleToggle.addEventListener('change', () => {
        rig.visible = visibleToggle.checked;
    });

    const nameLabel = document.createElement('span');
    nameLabel.className = 'rig-row-name';
    nameLabel.textContent = rig.name;

    const MODE_LABELS = { mesh: 'Mesh', skeleton: 'Skeleton', both: 'Both' };
    const MODE_CYCLE = ['mesh', 'skeleton', 'both'];

    const modeToggle = document.createElement('button');
    modeToggle.type = 'button';
    modeToggle.className = 'rig-mode-toggle';
    modeToggle.title = 'Cycle Mesh / Skeleton / Both';
    const renderModeLabel = () => { modeToggle.textContent = MODE_LABELS[rig.mode] ?? rig.mode; };
    renderModeLabel();
    modeToggle.addEventListener('click', () => {
        const next = MODE_CYCLE[(MODE_CYCLE.indexOf(rig.mode) + 1) % MODE_CYCLE.length];
        rig.mode = next;
        renderModeLabel();
    });

    header.append(visibleToggle, nameLabel, modeToggle);

    const positionRow = document.createElement('div');
    positionRow.className = 'rig-row-position';
    positionRow.append(makePositionField('x', rig), makePositionField('y', rig), makePositionField('z', rig));

    row.append(header, positionRow);
    return row;
}

/**
 * @param {HTMLElement} container - emptied and filled with one row per rig
 * @param {ReturnType<typeof import('./sceneRig.js').createRig>[]} rigs
 */
export function buildControlPanel(container, rigs) {
    container.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'rig-panel-header';

    const title = document.createElement('h3');
    title.textContent = 'Controls';

    const collapseToggle = document.createElement('button');
    collapseToggle.type = 'button';
    collapseToggle.className = 'rig-panel-collapse';
    collapseToggle.title = 'Collapse / expand';
    collapseToggle.textContent = '-';

    header.append(title, collapseToggle);
    container.appendChild(header);

    const content = document.createElement('div');
    content.className = 'rig-panel-content';
    content.appendChild(makeSpeedControl(rigs));
    rigs.forEach(rig => content.appendChild(makeRigRow(rig)));
    container.appendChild(content);

    collapseToggle.addEventListener('click', () => {
        const collapsed = content.style.display === 'none';
        content.style.display = collapsed ? '' : 'none';
        collapseToggle.textContent = collapsed ? '-' : '+';
    });
}
