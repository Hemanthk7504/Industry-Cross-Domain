// Explainable AI (XAI) Charts Controller (SHAP, LIME, PDP, ICE)
// Mission Control Dynamic Dark/Light Theme Aware

let shapChartInstance = null;
let limeChartInstance = null;
let pdpIceChartInstance = null;

function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
}

window.renderXAICharts = function(explainabilityData) {
    if (!explainabilityData) return;
    renderShapChart(explainabilityData.shap);
    renderLimeRules(explainabilityData.lime);
    renderPdpIceChart(explainabilityData.pdp_ice.torque);
};

function renderShapChart(shapData) {
    const canvas = document.getElementById('shapWaterfallChart');
    if (!canvas || !shapData) return;

    const dark = isDarkMode();
    const labels = shapData.shap_attributions.map(a => a.name);
    const values = shapData.shap_attributions.map(a => a.shap_value);
    const bgColors = values.map(v => v >= 0 ? 'rgba(251, 113, 133, 0.85)' : 'rgba(52, 211, 153, 0.85)');
    const borderColors = values.map(v => v >= 0 ? '#FB7185' : '#34D399');

    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    const ctx = canvas.getContext('2d');
    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Attribution (Δ Failure Risk)',
                data: values,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            return `SHAP Value: ${val >= 0 ? '+' : ''}${val.toFixed(4)} (${val >= 0 ? 'Increases Failure Risk' : 'Reduces Risk/Healthy'})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(15, 23, 42, 0.06)' },
                    ticks: { color: dark ? '#8A94B2' : '#64748B', font: { size: 11, family: 'Inter' } },
                    title: { display: true, text: 'SHAP Attribution Contribution', color: dark ? '#CBD5E1' : '#475569', font: { weight: '600', family: 'Inter' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: dark ? '#E6EAF5' : '#1E293B', font: { size: 12, weight: '500', family: 'Inter' } }
                }
            }
        }
    });

    const baseValEl = document.getElementById('shap_base_val');
    if (baseValEl) baseValEl.innerText = `Base Value E[f(x)] = ${shapData.base_value.toFixed(3)}`;
}

function renderLimeRules(limeData) {
    const container = document.getElementById('limeRulesContainer');
    if (!container || !limeData) return;

    container.innerHTML = limeData.rules.map(r => `
        <div class="flex items-center justify-between p-3 rounded-xl border transition-all ${
            r.weight > 0 
                ? 'bg-rose-500/10 border-rose-500/20 text-rose-300' 
                : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
        }">
            <div class="flex items-center gap-2.5">
                <i data-lucide="${r.weight > 0 ? 'trending-up' : 'trending-down'}" class="w-4 h-4 shrink-0 ${r.weight > 0 ? 'text-rose-400' : 'text-emerald-400'}"></i>
                <span class="text-xs font-mono font-medium text-[var(--text-main)]">${r.rule}</span>
            </div>
            <div class="flex items-center gap-2">
                <span class="text-xs font-mono font-bold ${r.weight > 0 ? 'text-rose-400' : 'text-emerald-400'}">
                    ${r.weight > 0 ? '+' : ''}${r.weight.toFixed(2)}
                </span>
                <span class="badge-subtle text-[10px] ${r.weight > 0 ? 'badge-critical' : 'badge-healthy'}">
                    ${r.supports}
                </span>
            </div>
        </div>
    `).join('');

    if (window.lucide) window.lucide.createIcons();
}

function renderPdpIceChart(pdpIceData) {
    const canvas = document.getElementById('pdpIceChart');
    if (!canvas || !pdpIceData) return;

    const dark = isDarkMode();
    const xLabels = pdpIceData.pdp_curve.map(p => p.x);
    const pdpY = pdpIceData.pdp_curve.map(p => p.pdp);
    const iceY = pdpIceData.ice_curve.map(p => p.ice);

    if (pdpIceChartInstance) {
        pdpIceChartInstance.destroy();
    }

    const datasets = [
        {
            label: `Current Asset ICE Curve (${pdpIceData.feature_name})`,
            data: iceY,
            borderColor: '#6366F1',
            backgroundColor: 'rgba(99, 102, 241, 0.12)',
            borderWidth: 3,
            tension: 0.25,
            pointRadius: 3,
            pointBackgroundColor: '#22D3EE'
        },
        {
            label: `Global Population PDP Average`,
            data: pdpY,
            borderColor: dark ? '#E6EAF5' : '#0F172A',
            borderWidth: 2.5,
            borderDash: [6, 4],
            tension: 0.25,
            pointRadius: 0
        }
    ];

    if (pdpIceData.ice_background_variants) {
        pdpIceData.ice_background_variants.forEach((variant, idx) => {
            datasets.push({
                label: `Cohort Asset ICE Trace #${idx + 1}`,
                data: variant.map(v => v.y),
                borderColor: dark ? 'rgba(255, 255, 255, 0.15)' : 'rgba(15, 23, 42, 0.18)',
                borderWidth: 1,
                borderDash: [2, 2],
                tension: 0.2,
                pointRadius: 0,
                fill: false
            });
        });
    }

    const ctx = canvas.getContext('2d');
    pdpIceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 14, font: { size: 11, family: 'Inter' }, color: dark ? '#CBD5E1' : '#475569' }
                },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            return `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}% Failure Probability`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(15, 23, 42, 0.06)' },
                    ticks: { color: dark ? '#8A94B2' : '#64748B', font: { family: 'JetBrains Mono', size: 10 } },
                    title: {
                        display: true,
                        text: `${pdpIceData.feature_name} (${pdpIceData.unit})`,
                        color: dark ? '#CBD5E1' : '#475569',
                        font: { weight: '600', family: 'Inter' }
                    }
                },
                y: {
                    min: 0,
                    max: 1.0,
                    grid: { color: dark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(15, 23, 42, 0.06)' },
                    ticks: {
                        color: dark ? '#8A94B2' : '#64748B',
                        font: { family: 'JetBrains Mono', size: 10 },
                        callback: function(val) { return (val * 100) + '%'; }
                    },
                    title: {
                        display: true,
                        text: 'Predicted Failure Risk P(Y=1)',
                        color: dark ? '#CBD5E1' : '#475569',
                        font: { weight: '600', family: 'Inter' }
                    }
                }
            }
        }
    });
}

window.selectPdpFeature = async function(featureKey) {
    try {
        const res = await fetch(`/api/v1/explain/pdp-ice?feature=${featureKey}`);
        if (!res.ok) throw new Error("Feature PDP fetch failed");
        const data = await res.json();
        renderPdpIceChart(data);
    } catch (e) {
        console.error("Error switching PDP feature:", e);
    }
};
