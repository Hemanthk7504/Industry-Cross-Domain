// Industrial Dashboard & Live Inference Execution Controller
// Mission Control Dynamic Dark/Light Theme Aware

let lastPredictionResult = null;

async function executeInference() {
    const btn = document.getElementById('btn-run-inference');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 mr-2 animate-spin inline-block"></i> Analyzing Cross-Domain Pipeline...`;
        if (window.lucide) window.lucide.createIcons();
    }

    try {
        const thresholdVal = parseFloat(document.getElementById('decision_threshold_input')?.value || 0.38);
        const assetId = document.getElementById('selected_asset_id')?.value || "CNC-SPINDLE-402";

        // Gather Target Machining Telemetry
        const targetSensors = {
            air_temperature: parseFloat(document.getElementById('target_air_temperature')?.value || 298.1),
            process_temperature: parseFloat(document.getElementById('target_process_temperature')?.value || 308.6),
            rotational_speed: parseFloat(document.getElementById('target_rotational_speed')?.value || 1540.0),
            torque: parseFloat(document.getElementById('target_torque')?.value || 40.2),
            tool_wear: parseFloat(document.getElementById('target_tool_wear')?.value || 85.0),
            vibration_index: parseFloat(document.getElementById('target_vibration_index')?.value || 2.1),
            acoustic_emission: parseFloat(document.getElementById('target_acoustic_emission')?.value || 62.0),
            oil_pressure: parseFloat(document.getElementById('target_oil_pressure')?.value || 4.5)
        };

        // Gather Source HVAC Telemetry
        const sourceHvacSensors = {
            indoor_temp: parseFloat(document.getElementById('hvac_indoor_temp')?.value || 22.5),
            return_air_temp: parseFloat(document.getElementById('hvac_return_air_temp')?.value || 24.2),
            supply_air_temp: parseFloat(document.getElementById('hvac_supply_air_temp')?.value || 14.5),
            chilled_water_supply_temp: parseFloat(document.getElementById('hvac_chilled_water_supply_temp')?.value || 6.8),
            chilled_water_return_temp: parseFloat(document.getElementById('hvac_chilled_water_return_temp')?.value || 12.4),
            fan_power: parseFloat(document.getElementById('hvac_fan_power')?.value || 7.2),
            compressor_vibration: parseFloat(document.getElementById('hvac_compressor_vibration')?.value || 1.8),
            air_flow_rate: parseFloat(document.getElementById('hvac_air_flow_rate')?.value || 3200.0),
            static_pressure: parseFloat(document.getElementById('hvac_static_pressure')?.value || 320.0),
            relative_humidity: parseFloat(document.getElementById('hvac_relative_humidity')?.value || 50.0)
        };

        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                asset_id: assetId,
                target_sensors: targetSensors,
                source_hvac_sensors: sourceHvacSensors,
                threshold: thresholdVal
            })
        });

        if (!response.ok) throw new Error("Inference execution failed");

        const result = await response.json();
        lastPredictionResult = result;
        updateUIWithPrediction(result);
        showToast("Inference generated with BiLSTM-BiGRU-VAE", "success");

    } catch (err) {
        console.error(err);
        showToast(`Inference error: ${err.message}`, "error");
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i data-lucide="zap" class="w-4 h-4 mr-2 text-amber-400 inline-block"></i> Run Two-Stage Inference`;
            if (window.lucide) window.lucide.createIcons();
        }
    }
}

function updateUIWithPrediction(data) {
    // 1. Prediction Banner & Probability
    const probEl = document.getElementById('res_failure_prob');
    const probBar = document.getElementById('res_prob_bar');
    const badgeEl = document.getElementById('res_prediction_badge');
    const confEl = document.getElementById('res_confidence');
    const titleEl = document.getElementById('res_prediction_title');

    const probPct = (data.failure_probability * 100).toFixed(1);
    const isFail = data.predicted_failure;

    if (probEl) probEl.innerText = `${probPct}%`;
    if (probBar) {
        probBar.style.width = `${probPct}%`;
        probBar.className = `h-2 rounded-full transition-all duration-500 ${
            isFail ? 'bg-rose-500 shadow-glow-rose' : 'bg-emerald-500'
        }`;
    }

    if (badgeEl) {
        badgeEl.className = `badge-subtle text-xs ${isFail ? 'badge-critical' : 'badge-healthy'}`;
        badgeEl.innerHTML = isFail 
            ? `<i data-lucide="alert-triangle" class="w-3.5 h-3.5 mr-1.5 inline-block text-rose-400"></i> ${data.prediction}`
            : `<i data-lucide="check-circle" class="w-3.5 h-3.5 mr-1.5 inline-block text-emerald-400"></i> ${data.prediction}`;
    }

    if (confEl) confEl.innerText = `${(data.confidence_score * 100).toFixed(1)}% Confidence`;
    if (titleEl) titleEl.innerText = data.diagnosis.mode_title;

    // 2. Stage 1 VAE Telemetry Card
    const vaeScoreEl = document.getElementById('res_vae_score');
    const vaeStatusEl = document.getElementById('res_vae_status');
    const vaeMseEl = document.getElementById('res_vae_mse');
    const vaeLatentList = document.getElementById('res_vae_latent_vector');

    if (vaeScoreEl) vaeScoreEl.innerText = data.stage1_outputs.vae_anomaly_score.toFixed(4);
    if (vaeStatusEl) {
        vaeStatusEl.innerText = data.stage1_outputs.vae_anomaly_status;
        vaeStatusEl.className = `badge-subtle text-xs ${
            data.stage1_outputs.vae_anomaly_score > 0.4 ? 'badge-warning' : 'badge-healthy'
        }`;
    }
    if (vaeMseEl) vaeMseEl.innerText = `Reconstruction MSE: ${data.stage1_outputs.reconstruction_loss_mse.toFixed(4)}`;

    if (vaeLatentList) {
        vaeLatentList.innerHTML = data.stage1_outputs.latent_embedding_vector.map((z, idx) => `
            <div class="px-2 py-1 bg-[var(--bg-elevated)] border border-[var(--border-color)] rounded-lg text-xs font-mono text-[var(--text-main)] flex justify-between">
                <span class="text-[var(--text-dim)]">z[${idx}]</span>
                <span class="font-semibold text-cyan-400">${z >= 0 ? '+' : ''}${z.toFixed(3)}</span>
            </div>
        `).join('');
    }

    // 3. Prescriptive Guidance & Work Order Card
    const rulHoursEl = document.getElementById('res_rul_hours');
    const rulCyclesEl = document.getElementById('res_rul_cycles');
    const rootCauseEl = document.getElementById('res_root_cause');
    const prescriptionsList = document.getElementById('res_prescriptions_list');
    const partsList = document.getElementById('res_parts_list');
    const btnDispatch = document.getElementById('btn_dispatch_work_order');

    if (rulHoursEl) rulHoursEl.innerText = `${data.diagnosis.rul_hours} hrs`;
    if (rulCyclesEl) rulCyclesEl.innerText = `${data.diagnosis.rul_cycles} cycles`;
    if (rootCauseEl) rootCauseEl.innerText = data.diagnosis.root_cause;

    if (prescriptionsList) {
        prescriptionsList.innerHTML = data.diagnosis.prescriptions.map(p => `
            <li class="flex items-start gap-2 text-xs text-[var(--text-muted)]">
                <i data-lucide="arrow-right" class="w-3.5 h-3.5 text-indigo-400 mt-0.5 shrink-0"></i>
                <span>${p}</span>
            </li>
        `).join('');
    }

    if (partsList) {
        if (data.diagnosis.parts_required && data.diagnosis.parts_required.length > 0) {
            partsList.innerHTML = data.diagnosis.parts_required.map(part => `
                <div class="flex items-center justify-between text-xs bg-[var(--bg-elevated)] border border-[var(--border-color)] p-2.5 rounded-xl">
                    <div>
                        <span class="font-semibold text-[var(--text-main)]">${part.sku}</span>
                        <p class="text-[var(--text-dim)]">${part.desc}</p>
                    </div>
                    <span class="badge-subtle badge-indigo">Qty: ${part.qty}</span>
                </div>
            `).join('');
        } else {
            partsList.innerHTML = `<p class="text-xs text-[var(--text-dim)] italic">No replacement spare parts required at this stage.</p>`;
        }
    }

    if (btnDispatch) {
        btnDispatch.style.display = data.diagnosis.work_order_required ? 'inline-flex' : 'none';
    }

    // 4. Update XAI tab if xai.js loaded
    if (window.renderXAICharts && data.explainability) {
        window.renderXAICharts(data.explainability);
    }

    if (window.lucide) window.lucide.createIcons();
}

// Dispatch Work Order Button Handler
async function dispatchCurrentWorkOrder() {
    if (!lastPredictionResult) {
        showToast("Please run an inference first", "warning");
        return;
    }
    
    try {
        const res = await fetch('/api/v1/work-orders/dispatch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                asset_id: lastPredictionResult.asset_id,
                diagnosis: lastPredictionResult.diagnosis
            })
        });
        
        if (!res.ok) throw new Error("Dispatch failed");
        const wo = await res.json();
        showToast(`Dispatched Work Order ${wo.work_order_id} to Field Crew`, "success");
        
        setTimeout(() => {
            window.location.href = `/work-orders?highlight=${wo.work_order_id}`;
        }, 1200);
    } catch (e) {
        showToast(`Dispatch failed: ${e.message}`, "error");
    }
}
