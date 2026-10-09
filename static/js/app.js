/**
 * CrossDomain AI — Global Application Animation & Interactive Engine
 * Linear / Vercel / Stripe Standard
 * Vanilla JS + GSAP + ScrollTrigger + Lucide + KaTeX
 */

(function () {
    'use strict';

    // =========================================================================
    // 1. THEME MANAGER (Dark-First "Mission Control" + LocalStorage Sync)
    // =========================================================================
    const THEME_KEY = 'cd_theme';
    let currentTheme = localStorage.getItem(THEME_KEY) || 'dark';

    function applyTheme(theme) {
        currentTheme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);

        // Update Theme Toggle Buttons
        document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
            const icon = btn.querySelector('[data-lucide]') || btn.querySelector('i');
            if (icon) {
                if (theme === 'dark') {
                    btn.setAttribute('title', 'Switch to Light Mode');
                    btn.setAttribute('aria-label', 'Switch to Light Mode');
                } else {
                    btn.setAttribute('title', 'Switch to Dark Mode');
                    btn.setAttribute('aria-label', 'Switch to Dark Mode');
                }
            }
        });

        // Trigger chart theme refresh if Chart.js is present
        if (window.Chart && window.Chart.instances) {
            Object.values(window.Chart.instances).forEach(inst => {
                try {
                    const isDark = theme === 'dark';
                    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(15, 23, 42, 0.08)';
                    const tickColor = isDark ? '#8A94B2' : '#475569';
                    const titleColor = isDark ? '#E2E8F0' : '#0F172A';

                    if (inst.options && inst.options.scales) {
                        Object.values(inst.options.scales).forEach(scale => {
                            if (scale.grid) scale.grid.color = gridColor;
                            if (scale.ticks) scale.ticks.color = tickColor;
                            if (scale.title) scale.title.color = titleColor;
                            if (scale.pointLabels) scale.pointLabels.color = titleColor;
                            if (scale.angleLines) scale.angleLines.color = isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(15, 23, 42, 0.12)';
                        });
                    }
                    if (inst.options && inst.options.plugins && inst.options.plugins.legend) {
                        if (inst.options.plugins.legend.labels) {
                            inst.options.plugins.legend.labels.color = titleColor;
                        }
                    }
                    inst.update('none');
                } catch (e) { }
            });
        }

        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    // Apply immediately to prevent flash
    applyTheme(currentTheme);

    window.toggleTheme = function () {
        const next = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        if (typeof showToast === 'function') {
            showToast(`Theme switched to ${next === 'dark' ? 'Dark Mission Control' : 'Clean Light Industrial'}`, 'info');
        }
    };

    // =========================================================================
    // 2. SCROLL & MOTION ENGINE
    // =========================================================================
    // Hardware-accelerated native smooth scroll managed by browser engine.

    // =========================================================================
    // 3. CURSOR SPOTLIGHT EFFECT (Mouse Glow on Glass Cards)
    // =========================================================================
    function initSpotlights() {
        const cards = document.querySelectorAll('.spotlight-card, .glass-card, .stat-card');
        cards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                card.style.setProperty('--mx', `${x}px`);
                card.style.setProperty('--my', `${y}px`);
            });
        });
    }

    // =========================================================================
    // 4. 3D TILT EFFECT ON HOVER (Max 6 deg + Glare)
    // =========================================================================
    function initTiltCards() {
        if (prefersReducedMotion) return;
        const tiltCards = document.querySelectorAll('.tilt-card');
        tiltCards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = ((y - centerY) / centerY) * -5.5;
                const rotateY = ((x - centerX) / centerX) * 5.5;

                card.style.transform = `perspective(1000px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateY(-2px)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)`;
                card.style.transition = 'transform 0.5s var(--ease-spring)';
            });

            card.addEventListener('mouseenter', () => {
                card.style.transition = 'transform 0.1s ease-out';
            });
        });
    }

    // =========================================================================
    // 5. METRIC COUNT-UP ANIMATION ENGINE
    // =========================================================================
    function initCountUps() {
        const elements = document.querySelectorAll('[data-countup]');
        if (!elements.length) return;

        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const el = entry.target;
                    obs.unobserve(el);

                    const targetVal = parseFloat(el.getAttribute('data-countup') || el.innerText.replace(/[^0-9.-]/g, ''));
                    const prefix = el.getAttribute('data-prefix') || '';
                    const suffix = el.getAttribute('data-suffix') || '';
                    const decimals = parseInt(el.getAttribute('data-decimals') || (el.getAttribute('data-countup')?.includes('.') ? el.getAttribute('data-countup').split('.')[1].length : 0));
                    const duration = parseFloat(el.getAttribute('data-duration') || 1.6);

                    if (isNaN(targetVal)) return;

                    const obj = { val: 0 };
                    if (window.gsap && !prefersReducedMotion) {
                        gsap.to(obj, {
                            val: targetVal,
                            duration: duration,
                            ease: 'power3.out',
                            onUpdate: () => {
                                el.innerText = `${prefix}${obj.val.toFixed(decimals)}${suffix}`;
                            }
                        });
                    } else {
                        el.innerText = `${prefix}${targetVal.toFixed(decimals)}${suffix}`;
                    }
                }
            });
        }, { threshold: 0.15 });

        elements.forEach(el => observer.observe(el));
    }

    // =========================================================================
    // 6. GSAP STAGGERED REVEALS (Page Load & Scroll)
    // =========================================================================
    function initGsapReveals() {
        if (!window.gsap || prefersReducedMotion) return;

        // Staggered Cards on Load
        const staggerCards = document.querySelectorAll('.stagger-card');
        if (staggerCards.length > 0) {
            gsap.from(staggerCards, {
                opacity: 0,
                y: 18,
                duration: 0.65,
                stagger: 0.06,
                ease: 'power3.out',
                clearProps: 'all'
            });
        }

        // Section reveals with ScrollTrigger
        if (window.ScrollTrigger) {
            const revealSections = document.querySelectorAll('.reveal-on-scroll');
            revealSections.forEach(sec => {
                gsap.from(sec, {
                    scrollTrigger: {
                        trigger: sec,
                        start: 'top 85%',
                    },
                    opacity: 0,
                    y: 24,
                    duration: 0.75,
                    ease: 'power3.out',
                    clearProps: 'opacity,transform'
                });
            });
        }
    }

    // =========================================================================
    // 7. SIDEBAR CONTROLLER (Collapsible + LocalStorage Memory)
    // =========================================================================
    const SIDEBAR_KEY = 'cd_sidebar_collapsed';

    window.toggleSidebar = function () {
        const isCollapsed = document.body.classList.toggle('sidebar-collapsed');
        localStorage.setItem(SIDEBAR_KEY, isCollapsed ? 'true' : 'false');
    };

    // Restore sidebar state
    if (localStorage.getItem(SIDEBAR_KEY) === 'true') {
        document.body.classList.add('sidebar-collapsed');
    }

    // Mobile Sidebar Drawer Toggle
    window.toggleMobileSidebar = function () {
        const aside = document.getElementById('app-sidebar');
        const overlay = document.getElementById('mobile-sidebar-overlay');
        if (aside && overlay) {
            const isHidden = aside.classList.contains('-translate-x-full');
            if (isHidden) {
                aside.classList.remove('-translate-x-full');
                overlay.classList.remove('hidden');
            } else {
                aside.classList.add('-translate-x-full');
                overlay.classList.add('hidden');
            }
        }
    };

    // =========================================================================
    // 8. COMMAND PALETTE (Cmd+K / Ctrl+K)
    // =========================================================================
    const COMMAND_ROUTES = [
        { title: 'Home Architecture', desc: 'Hybrid Two-Stage VAE & Cross-Domain Flow', url: '/', icon: 'home', badge: 'Core' },
        { title: 'Plant Fleet Overview', desc: 'Live Telemetry & Fleet Availability', url: '/dashboard', icon: 'layout-dashboard', badge: 'Fleet' },
        { title: 'Datasets Studio', desc: 'HVAC & Machinery Sensor Telemetry Hub', url: '/datasets', icon: 'database', badge: 'Data' },
        { title: 'Live Inference Lab', desc: 'Two-Stage VAE -> BiLSTM Real-Time Prediction', url: '/live-prediction', icon: 'zap', badge: 'Model' },
        { title: 'Digital Twin Stream', desc: 'Real-Time Sensor Telemetry Stream (1.0 Hz)', url: '/digital-twin', icon: 'activity', badge: 'Live' },
        { title: 'Batch CSV Ingestion', desc: 'Enterprise Multi-Asset CSV Batch Analytics', url: '/batch-analysis', icon: 'file-spreadsheet', badge: 'Batch' },
        { title: 'Prescriptive Orders', desc: 'Automated CMMS Work Order Generation', url: '/work-orders', icon: 'clipboard-list', badge: 'Ops' },
        { title: 'Enterprise Companies', desc: 'Multi-Tenant Company Profiles & Thresholds', url: '/company-profiles', icon: 'building-2', badge: 'Tenants' },
        { title: 'Training & Metrics', desc: 'Convergence Curves & Validation Metrics', url: '/model-metrics', icon: 'line-chart', badge: 'Metrics' },
        { title: 'Retraining Studio', desc: 'Adaptive Hyperparameter Engine & Retraining', url: '/training', icon: 'cpu', badge: 'Train' },
        { title: 'Stage 1: Anomaly Studio', desc: '8 Unsupervised Anomaly Benchmark Models', url: '/stage1-benchmark', icon: 'pie-chart', badge: 'Bench' },
        { title: 'Stage 2: Leaderboard', desc: 'BiLSTM-BiGRU-VAE Champion Benchmark (98.7%)', url: '/stage2-benchmark', icon: 'trophy', badge: 'Champion' },
        { title: 'Quad-XAI Diagnostics', desc: 'SHAP, LIME, PDP, ICE & Counterfactual Lab', url: '/explainability', icon: 'sparkles', badge: 'XAI' },
        { title: 'Threshold & Ablation', desc: 'Safety-Critical Threshold & Ablation Study', url: '/threshold-ablation', icon: 'sliders', badge: 'Safety' },
        { title: 'Audit & Drift Ledger', desc: 'SHA-256 Tamper-Proof Audit & KS-Drift Logs', url: '/audit-log', icon: 'shield-check', badge: 'Audit' },
        { title: 'REST API Specs', desc: 'Interactive OpenAPI / Swagger Documentation', url: '/api-docs', icon: 'code-2', badge: 'API' }
    ];

    window.openCommandPalette = function () {
        const backdrop = document.getElementById('cmd-backdrop');
        const input = document.getElementById('cmd-input');
        if (backdrop) {
            backdrop.classList.add('active');
            renderCommandResults('');
            if (input) {
                input.value = '';
                setTimeout(() => input.focus(), 60);
            }
        }
    };

    window.closeCommandPalette = function () {
        const backdrop = document.getElementById('cmd-backdrop');
        if (backdrop) backdrop.classList.remove('active');
    };

    function renderCommandResults(query) {
        const container = document.getElementById('cmd-results');
        if (!container) return;

        const filtered = COMMAND_ROUTES.filter(item =>
            item.title.toLowerCase().includes(query.toLowerCase()) ||
            item.desc.toLowerCase().includes(query.toLowerCase()) ||
            item.badge.toLowerCase().includes(query.toLowerCase())
        );

        if (!filtered.length) {
            container.innerHTML = `
                <div class="p-8 text-center text-slate-500">
                    <p class="text-sm font-medium">No results found for "${query}"</p>
                    <p class="text-xs text-slate-400 mt-1">Try searching for "Inference", "Metrics", or "XAI"</p>
                </div>
            `;
            return;
        }

        container.innerHTML = filtered.map((item, idx) => `
            <a href="${item.url}" class="cmd-item ${idx === 0 ? 'selected' : ''}">
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center">
                        <i data-lucide="${item.icon}" class="w-4 h-4"></i>
                    </div>
                    <div>
                        <div class="text-sm font-semibold text-slate-200">${item.title}</div>
                        <div class="text-xs text-slate-400">${item.desc}</div>
                    </div>
                </div>
                <span class="badge-subtle badge-indigo text-[10px]">${item.badge}</span>
            </a>
        `).join('');

        if (window.lucide) window.lucide.createIcons();
    }

    // Global Keyboard Shortcuts (Cmd+K, Ctrl+K, Escape)
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            const backdrop = document.getElementById('cmd-backdrop');
            if (backdrop && backdrop.classList.contains('active')) {
                closeCommandPalette();
            } else {
                openCommandPalette();
            }
        }
        if (e.key === 'Escape') {
            closeCommandPalette();
            if (window.closeAssetDrawer) window.closeAssetDrawer();
        }
    });

    // =========================================================================
    // 9. ASSET DIAGNOSTICS SLIDE-OVER DRAWER
    // =========================================================================
    window.openAssetDrawer = function (assetId, assetName, healthScore, mode) {
        const backdrop = document.getElementById('asset-drawer-backdrop');
        if (!backdrop) return;

        backdrop.classList.add('active');

        const titleEl = document.getElementById('drawer-asset-title');
        const idEl = document.getElementById('drawer-asset-id');
        const scoreEl = document.getElementById('drawer-health-score');
        const modeEl = document.getElementById('drawer-failure-mode');

        if (titleEl) titleEl.innerText = assetName || assetId;
        if (idEl) idEl.innerText = assetId;
        if (scoreEl) scoreEl.innerText = `${healthScore || 92}%`;
        if (modeEl) modeEl.innerText = mode || 'Normal Operating Envelope';

        if (window.lucide) window.lucide.createIcons();
    };

    window.closeAssetDrawer = function () {
        const backdrop = document.getElementById('asset-drawer-backdrop');
        if (backdrop) backdrop.classList.remove('active');
    };

    // =========================================================================
    // 10. CHART.JS MISSION CONTROL GLOBAL THEME DEFAULTS
    // =========================================================================
    if (window.Chart) {
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.color = currentTheme === 'dark' ? '#8A94B2' : '#64748B';
        Chart.defaults.borderColor = currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.06)' : 'rgba(15, 23, 42, 0.07)';
        Chart.defaults.plugins.tooltip.backgroundColor = currentTheme === 'dark' ? 'rgba(14, 20, 36, 0.95)' : 'rgba(255, 255, 255, 0.95)';
        Chart.defaults.plugins.tooltip.titleColor = currentTheme === 'dark' ? '#FFFFFF' : '#0F172A';
        Chart.defaults.plugins.tooltip.bodyColor = currentTheme === 'dark' ? '#CBD5E1' : '#334155';
        Chart.defaults.plugins.tooltip.borderColor = currentTheme === 'dark' ? 'rgba(255, 255, 255, 0.12)' : 'rgba(15, 23, 42, 0.1)';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.padding = 10;
        Chart.defaults.plugins.tooltip.cornerRadius = 10;
        Chart.defaults.plugins.tooltip.boxPadding = 4;
    }

    // =========================================================================
    // 11. INITIALIZATION ON DOM READY
    // =========================================================================
    document.addEventListener('DOMContentLoaded', () => {
        // Init Lucide Icons
        if (window.lucide) {
            window.lucide.createIcons();
        }

        // Init Math with KaTeX
        if (window.renderMathInElement) {
            window.renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false }
                ],
                throwOnError: false
            });
        }

        // Init UI Modules
        initSpotlights();
        initTiltCards();
        initCountUps();
        initGsapReveals();

        // Bind Command Palette Search
        const cmdInput = document.getElementById('cmd-input');
        if (cmdInput) {
            cmdInput.addEventListener('input', (e) => {
                renderCommandResults(e.target.value);
            });
        }
    });

})();
