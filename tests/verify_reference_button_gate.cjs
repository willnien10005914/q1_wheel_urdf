// CPU-only DOM fixture: verify the same module shipped in the compiled viewer.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const root = path.resolve(__dirname, '..');
global.document = {addEventListener() {}};
function button(action) {
  return {dataset: {pose: action}, classList: {remove() {}, toggle() {}},
    setAttribute(name, value) {this[name] = value;}};
}
(async () => {
  const {updateActionButtons, actionStatus} = await import(pathToFileURL(path.join(root, 'web/button_feedback.mjs')));
  const buttons = ['stand', 'unbox', 'kneel', 'skate', 'slide', 'backflip'].map(button);
  const fixture = {querySelectorAll: () => buttons};
  const state = {pose: 'kneel', policies: {unbox: true, backflip: true},
    supported_actions: {stand: true, unbox: true, kneel: false, skate: false, slide: false, backflip: false},
    action_unavailable_reasons: {kneel: '目前是蹲姿，請先按 Stand up 起身並等待站穩。',
      backflip: '目前是蹲姿，請先按 Stand up 起身並等待站穩。'}};
  updateActionButtons(state, fixture);
  assert.equal(buttons[1].disabled, false);
  assert.equal(buttons[5].disabled, true);
  assert.equal(buttons[5].dataset.actionState, '請先起立');
  assert.match(buttons[5].title, /蹲姿/);
  state.pose = 'unbox';state.supported_actions.unbox = false;
  updateActionButtons(state, fixture);
  assert.equal(buttons[1]['aria-busy'], 'true');
  assert.equal(buttons[1].disabled, true);
  assert.equal(actionStatus('backflip', {pose: 'unbox_hold',last_completed_motion: 'backflip'}), '已完成');
  const module = fs.readFileSync(path.join(root, 'web/button_feedback.mjs'), 'utf8').replaceAll('export function ', 'function ');
  const bundle = fs.readFileSync(path.join(root, 'web/app.bundle.js'), 'utf8');
  assert.ok(bundle.startsWith('// button_feedback.mjs?v=20261025-slow-kneel\n' + module));
  assert.ok(bundle.includes('const recovery = document.getElementById("recovery-availability");'));
  console.log('REFERENCE_BUTTON_GATE_PASS');
})().catch(error => {console.error(error);process.exit(1);});
