(function () {
  "use strict";

  /* ===== Helpers ===== */

  function $(id) { return document.getElementById(id); }
  function num(id) { return Number($(id).value); }
  function val(id) { return $(id).value; }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `HTTP ${res.status}`);
    }
    return res.json();
  }

  /* ===== Health Check ===== */

  async function checkHealth() {
    const badge = $("serverStatus");
    try {
      const h = await api("/api/health");
      badge.className = "status-badge ok";
      badge.querySelector(".label").textContent = "Server Online";
    } catch {
      badge.className = "status-badge err";
      badge.querySelector(".label").textContent = "Server Offline";
    }
  }

  /* ===== Interactive Shaft-Coupling 3D Preview ===== */

  const previewState = { yaw: -0.55, pitch: 0.30, zoom: 1.0, dragging: false, lastX: 0, lastY: 0 };

  function rot3(p) {
    const cy = Math.cos(previewState.yaw), sy = Math.sin(previewState.yaw);
    const cp = Math.cos(previewState.pitch), sp = Math.sin(previewState.pitch);
    const x1 = p.x * cy - p.z * sy;
    const z1 = p.x * sy + p.z * cy;
    const y2 = p.y * cp - z1 * sp;
    const z2 = p.y * sp + z1 * cp;
    return { x: x1, y: y2, z: z2 };
  }

  function drawDomainPreview() {
    const canvas = $("domainCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const L = Math.max(num("meshLx"), 0.01);
    const Rc = Math.max(num("couplingRadius"), 0.003);
    const Rs = Math.min(Math.max(num("shaftRadius"), 0.003), Rc * 0.98);
    const iface = Math.min(Math.max(num("interfaceLength"), 0), L * 0.48);
    const shaftLen = Math.max(L * 0.34, Rc * 2.4);
    const total = L + 2 * shaftLen;
    const scale = Math.min(w * 0.72 / total, h * 0.58 / (2 * Rc)) * previewState.zoom;
    const cx = w * 0.50, cy0 = h * 0.50;

    function proj(p) {
      const q = rot3(p);
      const perspective = 1 / (1 + q.z * 0.08);
      return { x: cx + q.x * scale * perspective, y: cy0 - q.y * scale * perspective, z: q.z };
    }

    // Background grid / axes
    ctx.fillStyle = "#09111e";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgba(136,155,185,0.12)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = h * 0.16 + i * h * 0.068;
      ctx.beginPath(); ctx.moveTo(w * 0.08, y); ctx.lineTo(w * 0.92, y); ctx.stroke();
    }

    function cylinder(x0, x1, r, fill, stroke) {
      const seg = 36;
      const sideA = [], sideB = [];
      for (let i = 0; i <= seg; i++) {
        const a = i / seg * Math.PI * 2;
        sideA.push(proj({x:x0, y:r*Math.cos(a), z:r*Math.sin(a)}));
        sideB.push(proj({x:x1, y:r*Math.cos(a), z:r*Math.sin(a)}));
      }
      // body as quad strips, depth sorted
      const strips=[];
      for (let i=0;i<seg;i++) strips.push({i, z:(sideA[i].z+sideA[i+1].z+sideB[i].z+sideB[i+1].z)/4});
      strips.sort((a,b)=>a.z-b.z);
      for (const s of strips) {
        const i=s.i;
        ctx.beginPath();
        ctx.moveTo(sideA[i].x,sideA[i].y); ctx.lineTo(sideA[i+1].x,sideA[i+1].y);
        ctx.lineTo(sideB[i+1].x,sideB[i+1].y); ctx.lineTo(sideB[i].x,sideB[i].y); ctx.closePath();
        ctx.fillStyle=fill; ctx.fill();
      }
      ctx.strokeStyle=stroke; ctx.lineWidth=1.4;
      [sideA,sideB].forEach(ring=>{ctx.beginPath(); ring.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)); ctx.stroke();});
    }

    const xL0 = -L/2-shaftLen, xL1=-L/2;
    const xR0 = L/2, xR1=L/2+shaftLen;

    cylinder(xL0,xL1,Rs,"rgba(119,133,153,0.78)","#b3bdca");
    cylinder(-L/2,L/2,Rc,"rgba(70,130,220,0.55)","#72b5ff");
    if (iface > 0) {
      cylinder(-L/2,-L/2+iface,Rc,"rgba(76,220,163,0.62)","#70e3b4");
      cylinder(L/2-iface,L/2,Rc,"rgba(76,220,163,0.62)","#70e3b4");
    }
    cylinder(xR0,xR1,Rs,"rgba(119,133,153,0.78)","#b3bdca");

    function arrow3(origin, vector, color, label) {
      const p0=proj(origin), p1=proj({x:origin.x+vector.x,y:origin.y+vector.y,z:origin.z+vector.z});
      const dx=p1.x-p0.x, dy=p1.y-p0.y, len=Math.hypot(dx,dy)||1;
      const ux=dx/len, uy=dy/len;
      ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=3;
      ctx.beginPath(); ctx.moveTo(p0.x,p0.y); ctx.lineTo(p1.x,p1.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p1.x,p1.y);
      ctx.lineTo(p1.x-ux*12-uy*6,p1.y-uy*12+ux*6);
      ctx.lineTo(p1.x-ux*12+uy*6,p1.y-uy*12-ux*6); ctx.closePath(); ctx.fill();
      ctx.font="700 12px sans-serif"; ctx.fillText(label,p1.x+8,p1.y-8);
    }

    // Force arrow at right shaft end
    const F = num("loadMagnitude");
    if (Math.abs(F) > 1e-12) {
      const d = val("loadDirection");
      const s = Math.sign(F) || 1;
      const aLen = Math.max(Rc*2.2, L*0.18);
      const v = {x:0,y:0,z:0}; v[d]=s*aLen;
      arrow3({x:xR1,y:0,z:0},v,"#ff6b78",`${d.toUpperCase()} force`);
    }

    // Torque ring on loaded shaft end (visual axis follows selection)
    if ($("torqueEnabled").checked && Math.abs(num("torqueMagnitude")) > 1e-12) {
      const axis=val("torqueAxis"), sign=Math.sign(num("torqueMagnitude"))||1;
      const center={x:xR1-shaftLen*0.12,y:0,z:0};
      const rr=Rs*1.75, pts=[];
      for(let i=0;i<=48;i++){
        const a=sign*(i/48)*Math.PI*1.65;
        let p={...center};
        if(axis==="x"){p.y=rr*Math.cos(a);p.z=rr*Math.sin(a);}
        if(axis==="y"){p.x=center.x+rr*Math.cos(a);p.z=rr*Math.sin(a);}
        if(axis==="z"){p.x=center.x+rr*Math.cos(a);p.y=rr*Math.sin(a);}
        pts.push(proj(p));
      }
      ctx.strokeStyle="#ffc857"; ctx.lineWidth=3; ctx.beginPath();
      pts.forEach((q,i)=>i?ctx.lineTo(q.x,q.y):ctx.moveTo(q.x,q.y)); ctx.stroke();
      if(pts.length>2){
        const p0=pts[pts.length-3],p1=pts[pts.length-1],dx=p1.x-p0.x,dy=p1.y-p0.y,l=Math.hypot(dx,dy)||1,ux=dx/l,uy=dy/l;
        ctx.fillStyle="#ffc857";ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p1.x-ux*11-uy*6,p1.y-uy*11+ux*6);ctx.lineTo(p1.x-ux*11+uy*6,p1.y-uy*11-ux*6);ctx.closePath();ctx.fill();
        ctx.font="700 12px sans-serif";ctx.fillText(`${axis.toUpperCase()} torque`,p1.x+8,p1.y-6);
      }
    }

    // Fixed end marker
    const fixed=proj({x:xL0,y:0,z:0});
    ctx.fillStyle="#d5dde8"; ctx.font="700 12px sans-serif"; ctx.fillText("FIXED",fixed.x-22,fixed.y-18);

    // Axes mini-gizmo
    const o={x:48,y:h-42};
    const colors={x:"#ff6b78",y:"#63d69f",z:"#69a9ff"};
    [["x",{x:0.12,y:0,z:0}],["y",{x:0,y:0.12,z:0}],["z",{x:0,y:0,z:0.12}]].forEach(([a,v])=>{
      const q=rot3(v),l=32;ctx.strokeStyle=colors[a];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(o.x,o.y);ctx.lineTo(o.x+q.x*l,o.y-q.y*l);ctx.stroke();ctx.fillStyle=colors[a];ctx.font="700 10px sans-serif";ctx.fillText(a.toUpperCase(),o.x+q.x*l+4,o.y-q.y*l);
    });
  }

  function setup3DPreviewInteraction() {
    const canvas=$("domainCanvas");
    canvas.addEventListener("pointerdown",e=>{previewState.dragging=true;previewState.lastX=e.clientX;previewState.lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener("pointermove",e=>{if(!previewState.dragging)return; const dx=e.clientX-previewState.lastX,dy=e.clientY-previewState.lastY;previewState.yaw+=dx*0.008;previewState.pitch=Math.max(-1.25,Math.min(1.25,previewState.pitch+dy*0.008));previewState.lastX=e.clientX;previewState.lastY=e.clientY;drawDomainPreview();});
    canvas.addEventListener("pointerup",()=>{previewState.dragging=false;});
    canvas.addEventListener("pointercancel",()=>{previewState.dragging=false;});
    canvas.addEventListener("wheel",e=>{e.preventDefault();previewState.zoom=Math.max(0.65,Math.min(1.65,previewState.zoom*(e.deltaY>0?0.92:1.08)));drawDomainPreview();},{passive:false});
  }

  /* ===== Boundary Conditions ===== */

  let bcCount = 0;

  function addBoundaryCondition(axis, coordMode) {
    bcCount++;
    const id = bcCount;
    const container = $("bcList");
    const div = document.createElement("div");
    div.className = "bc-item";
    div.id = `bc-${id}`;
    div.innerHTML = `
      <span class="bc-label">Axis</span>
      <select class="bc-axis">
        <option value="x" ${axis === "x" ? "selected" : ""}>X</option>
        <option value="y" ${axis === "y" ? "selected" : ""}>Y</option>
        <option value="z" ${axis === "z" ? "selected" : ""}>Z</option>
      </select>
      <span class="bc-label">Pos</span>
      <select class="bc-coord">
        <option value="min" ${coordMode === "min" ? "selected" : ""}>Min</option>
        <option value="max" ${coordMode === "max" ? "selected" : ""}>Max</option>
      </select>
      <button class="btn-remove" type="button" title="Remove">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>`;
    div.querySelector(".btn-remove").addEventListener("click", () => div.remove());
    container.appendChild(div);
  }

  function collectBoundaryConditions() {
    const items = document.querySelectorAll(".bc-item");
    const bcs = [];
    items.forEach((item) => {
      bcs.push({
        axis: item.querySelector(".bc-axis").value,
        coord_mode: item.querySelector(".bc-coord").value,
      });
    });
    return bcs.length > 0 ? bcs : [{ axis: "x", coord_mode: "min" }];
  }

  /* ===== Load Type Toggle ===== */

  function setupLoadTypeToggle() {
    const sel = $("loadType");
    const dist = $("distributedFields");
    const pt = $("pointFields");
    const coordSel = $("loadCoordMode");
    const coordVal = $("loadCoordValueWrap");

    function sync() {
      const isPoint = sel.value === "point";
      dist.style.display = isPoint ? "none" : "";
      pt.style.display = isPoint ? "" : "none";
    }

    function syncCoord() {
      coordVal.style.display = coordSel.value === "value" ? "" : "none";
    }

    sel.addEventListener("change", sync);
    coordSel.addEventListener("change", syncCoord);
    sync();
    syncCoord();
  }

  /* ===== Intermediate Snapshot Toggle ===== */

  function setupSnapshotToggle() {
    const cb = $("saveIntermediate");
    const wrap = $("snapshotIntervalWrap");
    cb.addEventListener("change", () => {
      wrap.style.display = cb.checked ? "" : "none";
    });
    wrap.style.display = cb.checked ? "" : "none";
  }

  /* ===== Volume Fraction Slider ===== */

  function setupSlider() {
    const slider = $("optVolumeFraction");
    const display = $("vfDisplay");
    slider.addEventListener("input", () => {
      display.textContent = Number(slider.value).toFixed(2);
    });
  }

  /* ===== V6 Run Modes / Website Safety Limits ===== */

  const RUN_LIMITS = {
    quick: { nx: 8, ny: 6, nz: 6, iterations: 10, label: "Quick Preview" },
    normal: { nx: 20, ny: 15, nz: 15, iterations: 100, label: "Normal Run" },
  };

  function currentMeshStats() {
    const nx = Math.max(1, Math.floor(num("meshNx") || 1));
    const ny = Math.max(1, Math.floor(num("meshNy") || 1));
    const nz = Math.max(1, Math.floor(num("meshNz") || 1));
    return {
      nx, ny, nz,
      elements: nx * ny * nz,
      nodes: (nx + 1) * (ny + 1) * (nz + 1),
    };
  }

  function updateComputationSize() {
    const stats = currentMeshStats();
    $("currentElements").textContent = stats.elements.toLocaleString();
    $("currentNodes").textContent = stats.nodes.toLocaleString();
  }

  function setRunButtonsDisabled(disabled) {
    $("quickRunBtn").disabled = disabled;
    $("normalRunBtn").disabled = disabled;
  }

  function validateNormalLimits(payload) {
    const m = payload.mesh;
    const it = payload.optimizer.iterations;
    const lim = RUN_LIMITS.normal;
    const problems = [];
    if (m.nx > lim.nx) problems.push(`Elements X ${m.nx} > ${lim.nx}`);
    if (m.ny > lim.ny) problems.push(`Elements Y ${m.ny} > ${lim.ny}`);
    if (m.nz > lim.nz) problems.push(`Elements Z ${m.nz} > ${lim.nz}`);
    if (it > lim.iterations) problems.push(`Iterations ${it} > ${lim.iterations}`);
    if (problems.length) {
      throw new Error(`Normal Run exceeds website limits: ${problems.join(", ")}. Maximum is 20 × 15 × 15 and 100 iterations.`);
    }
  }

  function prepareRunPayload(mode) {
    const payload = buildPayload();
    payload.run_mode = mode;
    payload.requested_parameters = {
      mesh: { ...payload.mesh },
      iterations: payload.optimizer.iterations,
    };

    if (mode === "quick") {
      const lim = RUN_LIMITS.quick;
      payload.mesh.nx = Math.min(payload.mesh.nx, lim.nx);
      payload.mesh.ny = Math.min(payload.mesh.ny, lim.ny);
      payload.mesh.nz = Math.min(payload.mesh.nz, lim.nz);
      payload.optimizer.iterations = Math.min(payload.optimizer.iterations, lim.iterations);
    } else {
      validateNormalLimits(payload);
    }
    return payload;
  }

  /* ===== Build Payload ===== */

  function buildPayload() {
    const loadType = val("loadType");
    return {
      mesh: {
        Lx: num("meshLx"), Ly: num("meshLy"), Lz: num("meshLz"),
        nx: num("meshNx"), ny: num("meshNy"), nz: num("meshNz"),
      },
      geometry: {
        type: "cylindrical_coupling",
        shaft_radius: num("shaftRadius"),
        radius: num("couplingRadius"),
        interface_length: num("interfaceLength"),
      },
      material: {
        E_mod: num("matE"),
        nu: num("matNu"),
        Emin: 1e-3,
      },
      boundary_conditions: collectBoundaryConditions(),
      loading_condition: {
        type: loadType,
        axis: val("loadAxis"),
        coord_mode: val("loadCoordMode"),
        coord_value: num("loadCoordValue"),
        direction: val("loadDirection"),
        magnitude: num("loadMagnitude"),
        location: [num("pointX"), num("pointY"), num("pointZ")],
      },
      torque_condition: {
        enabled: $("torqueEnabled").checked,
        axis: val("torqueAxis"),
        coord_mode: val("torqueCoordMode"),
        magnitude: num("torqueMagnitude"),
      },
      optimizer: {
        method: val("optMethod"),
        nn_device: val("nnDevice"),
        iterations: num("optIterations"),
        volume_fraction: num("optVolumeFraction"),
        penalty: num("optPenalty"),
        initial_density: num("optInitialDensity"),
        filter_radius: num("optFilterRadius"),
      },
      visualization: { thresholds: [0.1, 0.3, 0.5] },
      save_intermediate: $("saveIntermediate").checked,
      snapshot_interval: num("snapshotInterval"),
    };
  }

  /* ===== Status Management ===== */

  function setStatus(state, label, detail) {
    const bar = $("statusBar");
    bar.className = "status-bar " + state;

    const icons = {
      "": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
      running: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>',
      completed: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
      error: '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    };

    const iconEl = $("statusIcon");
    iconEl.innerHTML = icons[state] || icons[""];
    iconEl.className = "status-icon" + (state === "running" ? " spinning" : "");

    $("statusLabel").textContent = label;
    $("statusDetail").textContent = detail || "";
  }

  function showProgress(current, total) {
    const wrap = $("progressWrap");
    wrap.style.display = "";
    const pct = Math.round((current / total) * 100);
    $("progressFill").style.width = pct + "%";
    $("progressText").textContent = `${current}/${total}`;
  }

  function hideProgress() {
    $("progressWrap").style.display = "none";
  }

  function showError(msg) {
    const box = $("errorBox");
    box.style.display = "";
    $("errorText").textContent = msg;
  }

  function hideError() {
    $("errorBox").style.display = "none";
  }

  /* ===== Render Results ===== */

  function axisLabel(a) {
    const m = { x: "X-axis", y: "Y-axis", z: "Z-axis", "0": "X-axis", "1": "Y-axis", "2": "Z-axis" };
    return m[a] || a;
  }

  function renderResults(result) {
    $("emptyState").style.display = "none";
    $("resultsContainer").style.display = "";

    lastTaskId = result.task_id || lastTaskId;
    $("stlExportStatus").innerHTML = "";
    $("stlDownloadLinks").innerHTML = "";

    const cfg = result.input_summary;
    const engineEl = $("engineSummary");
    engineEl.innerHTML = `
      <div class="summary-item"><div class="s-label">Selected Engine</div><div class="s-value">${result.engine_label || result.engine || "—"}</div></div>
      <div class="summary-item"><div class="s-label">Method ID</div><div class="s-value">${result.engine || "—"}</div></div>`;
    const notesEl = $("engineNotes");
    notesEl.innerHTML = (result.notes || []).map(n => `<div class="engine-note">${n}</div>`).join("");
    const summaryEl = $("inputSummary");
    const items = [];
    items.push({ label: "Run Mode", value: (cfg.run_mode || "normal") === "quick" ? "Quick Preview" : "Normal Run" });

    (cfg.boundary_conditions || []).forEach((bc, i) => {
      items.push({ label: `Fixed Face ${i + 1}`, value: `${axisLabel(bc.axis)} = ${bc.coord_mode}` });
    });

    const ld = cfg.loading_condition || {};
    items.push({ label: "Load Type", value: ld.type || "distributed" });
    items.push({ label: "Load Direction", value: axisLabel(ld.direction || "y") });
    items.push({ label: "Load Magnitude", value: `${ld.magnitude} N` });

    const mesh = cfg.mesh || {};
    items.push({ label: "Domain Size", value: `${mesh.Lx} x ${mesh.Ly} x ${mesh.Lz} m` });
    items.push({ label: "Mesh Resolution", value: `${mesh.nx} x ${mesh.ny} x ${mesh.nz}` });

    const opt = cfg.optimizer || {};
    items.push({ label: "Optimization Method", value: opt.method || "shaft_simp" });
    if ((opt.method || "") === "teacher_nn") items.push({ label: "NN Device", value: opt.nn_device || "auto" });
    items.push({ label: "Volume Fraction", value: opt.volume_fraction });
    items.push({ label: "Iterations", value: opt.iterations });
    items.push({ label: "Penalty", value: opt.penalty });
    items.push({ label: "Filter Radius", value: opt.filter_radius });

    summaryEl.innerHTML = items.map((it) =>
      `<div class="summary-item"><div class="s-label">${it.label}</div><div class="s-value">${it.value}</div></div>`
    ).join("");

    const m = result.metrics || {};
    const metricsEl = $("metricsGrid");
    const metrics = [
      { label: "Compliance Improvement", value: `${m.compliance_improvement_percent?.toFixed(1) || "—"}%`, good: true },
      { label: "Final Compliance", value: m.final_compliance?.toExponential(3) || "—" },
      { label: "Final Volume", value: m.final_volume?.toFixed(4) || "—" },
      { label: "Density Mean", value: m.density_mean?.toFixed(4) || "—" },
      { label: "Density Min", value: m.density_min?.toFixed(4) || "—" },
      { label: "Density Max", value: m.density_max?.toFixed(4) || "—" },
    ];
    metricsEl.innerHTML = metrics.map((mt) =>
      `<div class="metric-card${mt.good ? " good" : ""}"><div class="m-value">${mt.value}</div><div class="m-label">${mt.label}</div></div>`
    ).join("");

    const structEl = $("structureImages");
    structEl.innerHTML = "";
    (result.structure_image_urls || []).forEach((url) => {
      const img = document.createElement("img");
      img.src = `${url}?t=${Date.now()}`;
      img.alt = "Optimized structure";
      img.loading = "lazy";
      structEl.appendChild(img);
    });

    const histSection = $("iterHistorySection");
    const snapEl = $("snapshotImages");
    const snapshots = result.snapshot_urls || [];
    if (snapshots.length > 0) {
      histSection.style.display = "";
      snapEl.innerHTML = "";
      snapshots.forEach((s) => {
        const card = document.createElement("div");
        card.className = "snapshot-card";
        card.innerHTML = `<img src="${s.url}?t=${Date.now()}" alt="Iteration ${s.iteration}" loading="lazy"/><div class="snap-label">Iteration ${s.iteration}</div>`;
        snapEl.appendChild(card);
      });
    } else {
      histSection.style.display = "none";
    }

    $("imgConvergence").src = `${result.convergence_url}?t=${Date.now()}`;
    $("imgHistogram").src = `${result.histogram_url}?t=${Date.now()}`;

    const dlEl = $("downloadLinks");
    dlEl.innerHTML = "";
    (result.files || []).forEach((f) => {
      const a = document.createElement("a");
      a.href = f.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.className = "download-link";
      a.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> ${f.name}`;
      dlEl.appendChild(a);
    });
  }

  /* ===== STL Export (density field -> printable mesh) ===== */

  let lastTaskId = null;

  async function exportStl() {
    if (!lastTaskId) return;
    const btn = $("exportStlBtn");
    const statusEl = $("stlExportStatus");
    const threshold = Number(val("stlThreshold"));
    const smoothIterations = Math.max(0, Math.round(num("stlSmooth") || 0));

    btn.disabled = true;
    statusEl.innerHTML = `<div class="engine-note">Running marching cubes + mesh cleanup (threshold ${threshold}, smoothing ${smoothIterations} iterations)...</div>`;

    try {
      const data = await api(`/api/optimize/export_stl/${lastTaskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ threshold, smooth_iterations: smoothIterations }),
      });

      const watertightNote = data.is_watertight
        ? `Closed (watertight) mesh, ${(data.volume_m3 * 1e6).toFixed(2)} cm&sup3; enclosed -- should slice cleanly.`
        : "Warning: mesh is NOT watertight after cleanup. Inspect it in your slicer before printing, or try a different threshold / add smoothing.";
      statusEl.innerHTML = `<div class="engine-note">${data.n_vertices} vertices, ${data.n_faces} faces. ${watertightNote}</div>`;

      const dlEl = $("stlDownloadLinks");
      const a = document.createElement("a");
      a.href = `${data.download_url}?t=${Date.now()}`;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.className = "download-link";
      a.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> ${data.file_name}`;
      dlEl.prepend(a);
    } catch (err) {
      statusEl.innerHTML = `<div class="engine-note" style="color:#c0392b">Export failed: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  /* ===== Polling ===== */

  let pollTimer = null;

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function pollTask(taskId) {
    try {
      const data = await api(`/api/optimize/status/${taskId}`);

      if (data.status === "running") {
        const p = data.progress;
        if (p) {
          showProgress(p.current, p.total);
          setStatus("running", "Running...", `Iteration ${p.current} / ${p.total}`);
        } else {
          setStatus("running", "Running...", "Optimization in progress");
        }
      } else if (data.status === "completed") {
        stopPolling();
        hideProgress();
        setStatus("completed", "Optimization Complete", "Results are ready below");
        renderResults(data.result);
        setRunButtonsDisabled(false);
      } else if (data.status === "error") {
        stopPolling();
        hideProgress();
        setStatus("error", "Error", data.error || "Optimization failed");
        showError(data.error || "Unknown error occurred");
        setRunButtonsDisabled(false);
      }
    } catch (err) {
      stopPolling();
      hideProgress();
      setStatus("error", "Polling Error", err.message);
      showError("Lost connection to server: " + err.message);
      setRunButtonsDisabled(false);
    }
  }

  /* ===== Start Optimization ===== */

  async function startOptimization(mode = "normal") {
    stopPolling();
    hideError();
    hideProgress();

    $("resultsContainer").style.display = "none";
    $("emptyState").style.display = "none";
    $("iterHistorySection").style.display = "none";
    setRunButtonsDisabled(true);

    setStatus("running", mode === "quick" ? "Quick Preview" : "Normal Run", mode === "quick" ? "Applying quick safety caps and submitting" : "Validating full settings and submitting");

    try {
      const payload = prepareRunPayload(mode);
      const task = await api("/api/optimize/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      setStatus("running", "Queued", `Task ID: ${task.task_id}`);
      showProgress(0, payload.optimizer.iterations);

      await pollTask(task.task_id);
      pollTimer = setInterval(() => pollTask(task.task_id), 1500);

    } catch (err) {
      setStatus("error", "Submission Failed", err.message);
      showError(err.message);
      setRunButtonsDisabled(false);
    }
  }

  /* ===== Init ===== */

  function init() {
    addBoundaryCondition("x", "min");
    $("addBcBtn").addEventListener("click", () => addBoundaryCondition("x", "min"));
    $("exportStlBtn").addEventListener("click", exportStl);

    setupLoadTypeToggle();
    setupSlider();
    setupSnapshotToggle();

    const optMethod = $("optMethod");
    const nnDeviceWrap = $("nnDeviceWrap");
    const optMethodHint = $("optMethodHint");
    function syncOptimizationMethod() {
      const method = optMethod.value;
      nnDeviceWrap.style.display = method === "teacher_nn" ? "" : "none";
      const hints = {
        shaft_simp: "Uses our cylindrical shaft-coupling mask and combined force + torque workflow.",
        teacher_numpy: "Calls the teacher-provided NumPy FEM + SIMP files without changing them. Baseline geometry remains the teacher's rectangular mesh.",
        teacher_nn: "Calls the teacher-provided differentiable PyTorch FEM + NNOptimizer without changing them. Baseline geometry remains rectangular."
      };
      optMethodHint.textContent = hints[method] || "";
    }
    optMethod.addEventListener("change", syncOptimizationMethod);
    syncOptimizationMethod();

    const torqueToggle = $("torqueEnabled");
    const torqueFields = $("torqueFields");
    function syncTorqueFields() {
      torqueFields.style.opacity = torqueToggle.checked ? "1" : "0.45";
      torqueFields.querySelectorAll("input, select").forEach((el) => {
        el.disabled = !torqueToggle.checked;
      });
    }
    torqueToggle.addEventListener("change", syncTorqueFields);
    syncTorqueFields();

    ["meshLx", "meshLy", "meshLz", "shaftRadius", "couplingRadius", "interfaceLength",
      "loadMagnitude", "loadDirection", "torqueMagnitude", "torqueAxis"].forEach((id) => {
      const el = $(id);
      el.addEventListener("input", drawDomainPreview);
      el.addEventListener("change", drawDomainPreview);
    });
    torqueToggle.addEventListener("change", drawDomainPreview);
    ["meshNx", "meshNy", "meshNz", "optIterations"].forEach((id) => {
      const el = $(id);
      el.addEventListener("input", updateComputationSize);
      el.addEventListener("change", updateComputationSize);
    });
    updateComputationSize();
    setup3DPreviewInteraction();
    drawDomainPreview();

    $("quickRunBtn").addEventListener("click", () => startOptimization("quick"));
    $("normalRunBtn").addEventListener("click", () => startOptimization("normal"));

    checkHealth();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
