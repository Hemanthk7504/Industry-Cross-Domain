// Main UI utilities and toast notification system

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg text-sm font-medium transition-all transform duration-300 translate-y-2 opacity-0 ${
        type === 'success' ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' :
        type === 'error' ? 'bg-rose-50 text-rose-800 border border-rose-200' :
        type === 'warning' ? 'bg-amber-50 text-amber-800 border border-amber-200' :
        'bg-blue-50 text-blue-800 border border-blue-200'
    }`;
    
    const icon = type === 'success' ? 'fa-check-circle text-emerald-600' :
                 type === 'error' ? 'fa-triangle-exclamation text-rose-600' :
                 type === 'warning' ? 'fa-circle-exclamation text-amber-600' :
                 'fa-circle-info text-blue-600';
                 
    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Animate in
    requestAnimationFrame(() => {
        toast.classList.remove('translate-y-2', 'opacity-0');
    });
    
    // Auto remove after 3.5s
    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-2');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// 1-Click Scenario Preset Loader
async function loadScenarioPreset(scenarioId) {
    try {
        const res = await fetch(`/api/v1/scenarios/${scenarioId}`);
        if (!res.ok) throw new Error("Scenario fetch failed");
        const data = await res.json();
        
        // Fill target sensors
        for (const [key, val] of Object.entries(data.target_sensors)) {
            const input = document.getElementById(`target_${key}`);
            if (input) {
                input.value = val;
                // Dispatch event for UI sliders/counters
                input.dispatchEvent(new Event('input'));
            }
        }
        
        // Fill source HVAC sensors
        for (const [key, val] of Object.entries(data.source_hvac_sensors)) {
            const input = document.getElementById(`hvac_${key}`);
            if (input) {
                input.value = val;
                input.dispatchEvent(new Event('input'));
            }
        }
        
        showToast(`Loaded scenario preset: ${data.title}`, 'success');
        
        // Auto-run inference if toggle is checked or run button exists
        const autoRunBtn = document.getElementById('btn-run-inference');
        if (autoRunBtn) {
            autoRunBtn.click();
        }
    } catch (e) {
        showToast(`Failed to load preset: ${e.message}`, 'error');
    }
}

// Quick 1-click Demo Account Login Helper
function fillDemoLogin(email) {
    const emailField = document.getElementById('login_email');
    const passField = document.getElementById('login_password');
    if (emailField && passField) {
        emailField.value = email;
        passField.value = 'plant123';
        showToast(`Filled credentials for ${email}`, 'info');
        const form = document.getElementById('login-form');
        if (form) form.submit();
    }
}
