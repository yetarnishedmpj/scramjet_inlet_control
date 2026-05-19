const canvas = document.getElementById("inletCanvas");
const ctx = canvas.getContext("2d");

const controls = {
  mach: document.getElementById("mach"),
  altitude: document.getElementById("altitude"),
  manualAngle: document.getElementById("manualAngle"),
  actuatorSpeed: document.getElementById("actuatorSpeed"),
  rolloutFile: document.getElementById("rolloutFile"),
  episodeSelect: document.getElementById("episodeSelect"),
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
  rolloutData: null,
  currentEpisode: null,
  currentStep: 0,
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
  if (state.mode === "replay") {
    stepReplay();
    return;
  }
  syncInputs();
  const metrics = computeMetrics();
  const command = clamp(controllerCommand(metrics), -0.8, 0.8);
  const actuatorSpeed = Number(controls.actuatorSpeed.value);
  state.rampRate += actuatorSpeed * (command - state.rampRate);
  state.rampRate = clamp(state.rampRate, -0.75, 0.75);
  state.rampAngle = clamp(state.rampAngle + state.rampRate, 4, 18);
  update();
}

function stepReplay() {
  if (!state.rolloutData || state.currentEpisode === null) return;
  const episodeData = state.rolloutData[state.currentEpisode];
  if (state.currentStep >= episodeData.length) {
    state.playing = false;
    clearInterval(timer);
    document.getElementById("playButton").textContent = "Play";
    return;
  }
  const row = episodeData[state.currentStep];
  state.mach = row.mach;
  state.altitude = row.altitude_m;
  state.rampAngle = row.ramp_angle_deg;
  state.rampRate = row.angle_rate_deg_per_step || 0;
  state.currentStep++;
  update();
}

function syncInputs() {
  state.mach = Number(controls.mach.value);
  state.altitude = Number(controls.altitude.value);
  state.manualAngle = Number(controls.manualAngle.value);
}

function update() {
  if (state.mode !== "replay") {
    syncInputs();
    if (state.mode === "manual" && !state.playing) {
      state.rampAngle = state.manualAngle;
      state.rampRate = 0;
    }
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

  if (state.mode === "replay" && state.rolloutData && state.currentEpisode !== null) {
    const episodeData = state.rolloutData[state.currentEpisode];
    const row = episodeData[Math.min(state.currentStep, episodeData.length - 1)];
    readouts.shockError.textContent = row.shock_error.toFixed(3);
    readouts.pressureRecovery.textContent = row.pressure_recovery.toFixed(3);
    readouts.efficiency.textContent = row.efficiency.toFixed(3);
    readouts.unstart.textContent = row.unstart_probability.toFixed(3);
    const unstart = row.unstart_probability;
    readouts.unstart.className = unstart > 0.65 ? "risk-high" : unstart > 0.25 ? "risk-med" : "risk-low";
  } else {
    readouts.shockError.textContent = metrics.shockError.toFixed(3);
    readouts.pressureRecovery.textContent = metrics.pressureRecovery.toFixed(3);
    readouts.efficiency.textContent = metrics.efficiency.toFixed(3);
    readouts.unstart.textContent = metrics.unstart.toFixed(3);
    readouts.unstart.className =
      metrics.unstart > 0.65 ? "risk-high" : metrics.unstart > 0.25 ? "risk-med" : "risk-low";
  }

  readouts.mode.textContent =
    state.mode === "rl"
      ? "RL Policy"
      : state.mode === "replay"
        ? "Replay"
        : state.mode[0].toUpperCase() + state.mode.slice(1);
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

  ctx.globalCompositeOperation = "source-over";
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#060d16");
  gradient.addColorStop(0.5, "#0b1828");
  gradient.addColorStop(1, metrics.unstart > 0.65 ? "#3a1315" : "#0d2138");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  drawFlowLines(metrics);
  drawInlet(metrics);
  drawShock(metrics);
  drawLabels(metrics);
}

function drawFlowLines(metrics) {
  ctx.save();
  ctx.globalCompositeOperation = "screen";
  const compression = 18 + metrics.shockError * 30;
  
  for (let i = 0; i < 9; i += 1) {
    const y = 90 + i * 40;
    
    ctx.beginPath();
    ctx.moveTo(20, y);
    ctx.bezierCurveTo(280, y, 520, y + compression * Math.sin(i), 1020, y + (i - 4) * 2);
    
    const grad = ctx.createLinearGradient(20, 0, 1020, 0);
    grad.addColorStop(0, "rgba(56, 189, 248, 0.05)");
    grad.addColorStop(0.5, "rgba(56, 189, 248, 0.4)");
    grad.addColorStop(1, "rgba(56, 189, 248, 0.05)");
    
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#38bdf8";
    ctx.stroke();
  }
  ctx.restore();
}

function drawInlet(metrics) {
  const lipX = 830;
  const lipY = 190;
  const rampStartX = 230;
  const rampStartY = 430;
  const rampLength = 480;
  const angleRad = (state.rampAngle * Math.PI) / 180;
  const rampEndX = rampStartX + rampLength * Math.cos(angleRad);
  const rampEndY = rampStartY - rampLength * Math.sin(angleRad);

  ctx.save();
  
  const bodyGrad = ctx.createLinearGradient(120, 150, 1010, 196);
  bodyGrad.addColorStop(0, "#1e293b");
  bodyGrad.addColorStop(1, "#0f172a");

  ctx.fillStyle = bodyGrad;
  ctx.beginPath();
  ctx.moveTo(120, 150);
  ctx.lineTo(lipX, lipY);
  ctx.lineTo(1010, lipY + 18);
  ctx.lineTo(1010, lipY + 52);
  ctx.lineTo(lipX, lipY + 34);
  ctx.lineTo(120, 196);
  ctx.closePath();
  ctx.fill();
  
  ctx.strokeStyle = "rgba(56, 189, 248, 0.3)";
  ctx.lineWidth = 1;
  ctx.stroke();

  const rampGrad = ctx.createLinearGradient(rampStartX, rampStartY, rampEndX, rampEndY);
  rampGrad.addColorStop(0, "#334155");
  rampGrad.addColorStop(1, "#1e293b");

  ctx.fillStyle = rampGrad;
  ctx.beginPath();
  ctx.moveTo(rampStartX, rampStartY);
  ctx.lineTo(rampEndX, rampEndY);
  ctx.lineTo(rampEndX + 40, rampEndY + 32);
  ctx.lineTo(rampStartX + 40, rampStartY + 32);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 3;
  ctx.shadowBlur = 10;
  ctx.shadowColor = "#38bdf8";
  
  ctx.beginPath();
  ctx.moveTo(rampStartX, rampStartY);
  ctx.lineTo(rampEndX, rampEndY);
  ctx.stroke();

  ctx.fillStyle = metrics.unstart > 0.65 ? "#ef4444" : "#10b981";
  ctx.shadowBlur = 15;
  ctx.shadowColor = ctx.fillStyle;
  ctx.beginPath();
  ctx.arc(lipX, lipY, 6, 0, Math.PI * 2);
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
  ctx.globalCompositeOperation = "screen";
  
  const isHighRisk = metrics.unstart > 0.65;
  ctx.strokeStyle = isHighRisk ? "#ef4444" : "#f59e0b";
  ctx.lineWidth = 6;
  ctx.lineCap = "round";
  
  ctx.shadowBlur = 20;
  ctx.shadowColor = ctx.strokeStyle;
  
  ctx.beginPath();
  ctx.moveTo(shockBaseX, shockBaseY);
  ctx.lineTo(shockTopX, shockTopY);
  ctx.stroke();
  
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffffff";
  ctx.shadowBlur = 0;
  ctx.stroke();

  ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(shockTopX, shockTopY);
  ctx.lineTo(lipX, lipY);
  ctx.stroke();
  
  ctx.restore();
}

function drawLabels(metrics) {
  ctx.save();
  ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
  ctx.strokeStyle = "rgba(56, 189, 248, 0.3)";
  ctx.lineWidth = 1;
  ctx.shadowBlur = 10;
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  
  roundRect(34, 34, 260, 96, 12);
  ctx.fill();
  ctx.stroke();
  
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "700 24px Outfit, sans-serif";
  ctx.fillText(`Mach ${state.mach.toFixed(1)}`, 54, 70);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "400 14px Inter, sans-serif";
  ctx.fillText(`Altitude ${Math.round(state.altitude)} m`, 54, 95);
  ctx.fillText(`Density ${metrics.density.toFixed(3)} kg/m³`, 54, 116);

  ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
  roundRect(760, 390, 290, 116, 12);
  ctx.shadowBlur = 10;
  ctx.shadowColor = "rgba(0,0,0,0.5)";
  ctx.fill();
  ctx.stroke();
  
  ctx.shadowBlur = 0;
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "700 18px Outfit, sans-serif";
  ctx.fillText("Controller Response", 782, 426);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "400 14px Inter, sans-serif";
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
    if (state.mode === "replay") {
      state.currentStep = 0;
    }
    update();
  });
});

controls.rolloutFile.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    const csv = event.target.result;
    parseRolloutCSV(csv);
  };
  reader.readAsText(file);
});

controls.episodeSelect.addEventListener("change", (e) => {
  state.currentEpisode = e.target.value === "" ? null : e.target.value;
  state.currentStep = 0;
  update();
});

function parseRolloutCSV(csv) {
  const lines = csv.split("\n");
  const headers = lines[0].split(",");
  const data = {};
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header.trim()] = parseFloat(values[index]);
    });
    const ep = row.episode;
    if (!data[ep]) data[ep] = [];
    data[ep].push(row);
  }
  state.rolloutData = data;
  const episodes = Object.keys(data);
  controls.episodeSelect.innerHTML = episodes.map((ep) => `<option value="${ep}">Episode ${ep}</option>`).join("");
  controls.episodeSelect.disabled = false;
  state.currentEpisode = episodes[0];
  state.currentStep = 0;
  update();
}

Object.values(controls).forEach((control) => {
  if (control.type === "range" || control.type === "select-one") {
    control.addEventListener("input", update);
  }
});
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
