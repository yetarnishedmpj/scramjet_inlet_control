const canvas = document.getElementById("inletCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  mach: document.getElementById("mach"),
  altitude: document.getElementById("altitude"),
  manualAngle: document.getElementById("manualAngle"),
  actuatorSpeed: document.getElementById("actuatorSpeed"),
};

const readouts = {
  mach: document.getElementById("machReadout"),
  altitude: document.getElementById("altitudeReadout"),
  manualAngle: document.getElementById("manualAngleReadout"),
  speed: document.getElementById("speedReadout"),
  rampAngle: document.getElementById("rampAngleValue"),
  targetAngle: document.getElementById("targetAngleValue"),
  shockError: document.getElementById("shockErrorValue"),
  pressureRecovery: document.getElementById("pressureRecoveryValue"),
  efficiency: document.getElementById("efficiencyValue"),
  unstart: document.getElementById("unstartValue"),
  mode: document.getElementById("modeLabel"),
};

const state = {
  mode: "manual",
  mach: 6,
  altitude: 20000,
  manualAngle: 10,
  rampAngle: 10,
  rampRate: 0,
  playing: false,
};

let timer = null;

function densityAtAltitude(altitude) {
  return 1.225 * Math.exp(-altitude / 8500);
}

function targetAngle(mach) {
  return clamp(6 + 1.6 * (mach - 4), 4, 18);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function computeMetrics() {
  const target = targetAngle(state.mach);
  const shockErrorSigned = (state.rampAngle - target) / 8;
  const shockError = Math.abs(shockErrorSigned);
  const density = densityAtAltitude(state.altitude);
  const pressureRecovery = clamp(0.94 - 0.16 * shockError - 0.018 * (state.mach - 4), 0.1, 1);
  const unstart = sigmoid((shockError - 0.65) * 10 + (state.mach - 7.3) * 0.55);
  const efficiency = clamp(1 - shockError - 0.55 * unstart, 0, 1);
  const shockX = clamp(0.72 - 0.045 * (state.rampAngle - target), 0.15, 0.95);
  const shockSlope = 0.42 + 0.035 * (state.mach - 4);
  return { target, shockErrorSigned, shockError, density, pressureRecovery, unstart, efficiency, shockX, shockSlope };
}

function controllerCommand(metrics) {
  if (state.mode === "manual") return state.manualAngle - state.rampAngle;
  if (state.mode === "baseline") return metrics.target - state.rampAngle;

  const shockCorrection = -metrics.shockErrorSigned * 5.2;
  const unstartCorrection = metrics.unstart > 0.35 ? -1.4 * metrics.unstart : 0;
  const recoveryBias = metrics.pressureRecovery < 0.82 ? 0.35 : 0;
  return shockCorrection + unstartCorrection + recoveryBias;
}

function stepSimulation() {
  syncInputs();
  const metrics = computeMetrics();
  const command = clamp(controllerCommand(metrics), -0.8, 0.8);
  const actuatorSpeed = Number(controls.actuatorSpeed.value);
  state.rampRate += actuatorSpeed * (command - state.rampRate);
  state.rampRate = clamp(state.rampRate, -0.75, 0.75);
  state.rampAngle = clamp(state.rampAngle + state.rampRate, 4, 18);
  update();
}

function syncInputs() {
  state.mach = Number(controls.mach.value);
  state.altitude = Number(controls.altitude.value);
  state.manualAngle = Number(controls.manualAngle.value);
}

function update() {
  syncInputs();
  if (state.mode === "manual" && !state.playing) {
    state.rampAngle = state.manualAngle;
    state.rampRate = 0;
  }
  const metrics = computeMetrics();
  updateReadouts(metrics);
  updateEffects(metrics);
  draw(metrics);
}

function updateReadouts(metrics) {
  readouts.mach.textContent = state.mach.toFixed(1);
  readouts.altitude.textContent = `${Math.round(state.altitude)} m`;
  readouts.manualAngle.textContent = `${state.manualAngle.toFixed(1)} deg`;
  readouts.speed.textContent = Number(controls.actuatorSpeed.value).toFixed(2);
  readouts.rampAngle.textContent = `${state.rampAngle.toFixed(1)} deg`;
  readouts.targetAngle.textContent = `${metrics.target.toFixed(1)} deg`;
  readouts.shockError.textContent = metrics.shockError.toFixed(3);
  readouts.pressureRecovery.textContent = metrics.pressureRecovery.toFixed(3);
  readouts.efficiency.textContent = metrics.efficiency.toFixed(3);
  readouts.unstart.textContent = metrics.unstart.toFixed(3);
  readouts.unstart.className = metrics.unstart > 0.65 ? "risk-high" : metrics.unstart > 0.25 ? "risk-med" : "risk-low";
  readouts.mode.textContent = state.mode === "rl" ? "RL Policy" : state.mode[0].toUpperCase() + state.mode.slice(1);
}

function updateEffects(metrics) {
  const effects = [
    `Higher Mach pushes the target ramp angle upward. Current target is ${metrics.target.toFixed(1)} deg.`,
    `Altitude changes air density. Current density estimate is ${metrics.density.toFixed(3)} kg/m^3.`,
    `Ramp angle moves the oblique shock. Current shock error is ${metrics.shockError.toFixed(3)}.`,
    `Large shock error increases unstart risk. Current unstart risk is ${metrics.unstart.toFixed(3)}.`,
    `Actuator speed controls how quickly the inlet can respond to changing conditions.`,
  ];
  document.getElementById("effectList").innerHTML = effects.map((item) => `<li>${item}</li>`).join("");
}

function draw(metrics) {
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#d9edf8");
  gradient.addColorStop(0.6, "#f4efe3");
  gradient.addColorStop(1, metrics.unstart > 0.65 ? "#f4b4ad" : "#d8eadf");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  drawFlowLines(metrics);
  drawInlet(metrics);
  drawShock(metrics);
  drawLabels(metrics);
}

function drawFlowLines(metrics) {
  ctx.save();
  ctx.strokeStyle = "rgba(16, 83, 122, 0.35)";
  ctx.lineWidth = 2;
  const compression = 18 + metrics.shockError * 30;
  for (let i = 0; i < 9; i += 1) {
    const y = 90 + i * 40;
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.bezierCurveTo(280, y, 520, y + compression * Math.sin(i), 1020, y + (i - 4) * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawInlet(metrics) {
  const lipX = 830;
  const lipY = 190;
  const lowerY = 395;
  const rampStartX = 230;
  const rampStartY = 430;
  const rampLength = 480;
  const angleRad = (state.rampAngle * Math.PI) / 180;
  const rampEndX = rampStartX + rampLength * Math.cos(angleRad);
  const rampEndY = rampStartY - rampLength * Math.sin(angleRad);

  ctx.save();
  ctx.fillStyle = "#243746";
  ctx.beginPath();
  ctx.moveTo(120, 150);
  ctx.lineTo(lipX, lipY);
  ctx.lineTo(1010, lipY + 18);
  ctx.lineTo(1010, lipY + 52);
  ctx.lineTo(lipX, lipY + 34);
  ctx.lineTo(120, 196);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#314a5c";
  ctx.beginPath();
  ctx.moveTo(rampStartX, rampStartY);
  ctx.lineTo(rampEndX, rampEndY);
  ctx.lineTo(rampEndX + 40, rampEndY + 32);
  ctx.lineTo(rampStartX + 40, rampStartY + 32);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#0f2535";
  ctx.lineWidth = 4;
  ctx.stroke();

  ctx.fillStyle = metrics.unstart > 0.65 ? "#b3261e" : "#1d8a57";
  ctx.beginPath();
  ctx.arc(lipX, lipY, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawShock(metrics) {
  const lipX = 830;
  const lipY = 190;
  const shockBaseX = 180 + metrics.shockX * 720;
  const shockBaseY = 425;
  const shockTopX = shockBaseX + metrics.shockSlope * 230;
  const shockTopY = 150;
  ctx.save();
  ctx.strokeStyle = metrics.unstart > 0.65 ? "#b3261e" : "#bd6a15";
  ctx.lineWidth = 5;
  ctx.setLineDash([12, 8]);
  ctx.beginPath();
  ctx.moveTo(shockBaseX, shockBaseY);
  ctx.lineTo(shockTopX, shockTopY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(shockTopX, shockTopY);
  ctx.lineTo(lipX, lipY);
  ctx.stroke();
  ctx.restore();
}

function drawLabels(metrics) {
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.86)";
  ctx.strokeStyle = "rgba(15,37,53,0.15)";
  ctx.lineWidth = 1;
  roundRect(34, 34, 260, 92, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#16202a";
  ctx.font = "700 22px Segoe UI";
  ctx.fillText(`Mach ${state.mach.toFixed(1)}`, 54, 68);
  ctx.font = "15px Segoe UI";
  ctx.fillText(`Altitude ${Math.round(state.altitude)} m`, 54, 95);
  ctx.fillText(`Density ${metrics.density.toFixed(3)} kg/m^3`, 54, 116);

  ctx.fillStyle = "rgba(255,255,255,0.86)";
  roundRect(760, 390, 290, 112, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#16202a";
  ctx.font = "700 18px Segoe UI";
  ctx.fillText("Controller Response", 782, 424);
  ctx.font = "15px Segoe UI";
  ctx.fillText(`Ramp rate ${state.rampRate.toFixed(2)} deg/step`, 782, 452);
  ctx.fillText(`Pressure recovery ${metrics.pressureRecovery.toFixed(3)}`, 782, 475);
  ctx.fillText(`Unstart risk ${metrics.unstart.toFixed(3)}`, 782, 498);
  ctx.restore();
}

function roundRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-mode]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.mode = button.dataset.mode;
    state.rampRate = 0;
    update();
  });
});

Object.values(controls).forEach((control) => control.addEventListener("input", update));
document.getElementById("stepButton").addEventListener("click", stepSimulation);
document.getElementById("resetButton").addEventListener("click", () => {
  state.rampAngle = Number(controls.manualAngle.value);
  state.rampRate = 0;
  update();
});
document.getElementById("playButton").addEventListener("click", (event) => {
  state.playing = !state.playing;
  event.target.textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    timer = setInterval(stepSimulation, 160);
  } else {
    clearInterval(timer);
  }
});

update();
