// One-shot feedback and live execution status are separate: these are commands,
// not toggle switches. Completion never leaves an action looking enabled/selected.
export function actionStatus(action, state) {
  const pose = state.pose;
  if (action === "kneel") return ["prepare_kneel", "kneeling"].includes(pose) ? "執行中" : pose === "kneel" ? "已跪穩" : "";
  if (action === "unbox") return pose === "unbox" ? "執行中" : pose === "unbox_hold" ? "已站穩" : "";
  if (action === "backflip") return pose === "prepare_backflip" ? "準備中" : pose === "backflip" ? (state.backflip?.success ? "已完成" : "執行中") : "";
  if (["skate", "slide"].includes(action)) return pose === action ? "執行中" : "";
  if (action.startsWith("recovery_")) return pose === action ? "執行中" : "";
  if (["supine", "prone"].includes(action)) return pose === action ? "目前姿勢" : "";
  return "";
}
export function updateActionButtons(state = {}, root = document) {
  root.querySelectorAll("[data-pose]").forEach((button) => {
    const action = button.dataset.pose;
    const pending = button.dataset.pending === "true";
    const status = pending ? "送出中" : actionStatus(action, state);
    const running = ["執行中", "準備中"].includes(status);
    const unavailable = action === "backflip" ? !state.policies?.backflip
      : action === "unbox" ? state.policies?.unbox === false
      : action.startsWith("recovery_") ? !state.policies?.[action.slice(9)] : false;
    button.disabled = pending || unavailable;
    button.dataset.actionState = status;
    button.classList.remove("active-pose");
    button.classList.toggle("action-running", running);
    button.classList.toggle("action-pending", pending);
    button.setAttribute("aria-busy", String(pending || running));
  });
}
export function animateButton(button) {
  if (!button || button.disabled) return;
  button.classList.remove("press-feedback");
  void button.offsetWidth;
  button.classList.add("press-feedback");
  clearTimeout(button._pressFeedbackTimer);
  button._pressFeedbackTimer = setTimeout(() => button.classList.remove("press-feedback"), 360);
}
document.addEventListener("click", (event) => animateButton(event.target.closest?.("button, .file-btn")), true);
