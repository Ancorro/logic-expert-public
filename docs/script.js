// Tab-switching functionality for Architecture Section
document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupFuzzyCalculator();
    setupRoutingSimulator();
    setupCitationCopy();
    setupCitationLinksNoScroll();
});

function setupCitationLinksNoScroll() {
    const citationLinks = document.querySelectorAll('.citation-link');

    citationLinks.forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
        });
    });
}

function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;

            // Update active states for buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update active states for panes
            tabPanes.forEach(pane => {
                if (pane.id === targetTab) {
                    pane.classList.add('active');
                } else {
                    pane.classList.remove('active');
                }
            });
        });
    });
}

// Fuzzy Logic Gate Calculator
function setupFuzzyCalculator() {
    const sliderT1 = document.getElementById('slider-t1');
    const sliderT2 = document.getElementById('slider-t2');
    
    const valueT1 = document.getElementById('value-t1');
    const valueT2 = document.getElementById('value-t2');
    
    const fillAnd = document.getElementById('fill-and');
    const fillOr = document.getElementById('fill-or');
    const fillNot1 = document.getElementById('fill-not-1');
    const fillNot2 = document.getElementById('fill-not-2');
    
    const valAnd = document.getElementById('val-and');
    const valOr = document.getElementById('val-or');
    const valNot1 = document.getElementById('val-not-1');
    const valNot2 = document.getElementById('val-not-2');

    function updateCalculator() {
        const t1 = parseFloat(sliderT1.value);
        const t2 = parseFloat(sliderT2.value);

        valueT1.textContent = t1.toFixed(2);
        valueT2.textContent = t2.toFixed(2);

        // Fuzzy logic computations (clipped between 0 and 1, sliders are already restricted)
        const andResult = t1 * t2;
        const orResult = t1 + t2 - (t1 * t2);
        const not1Result = 1.0 - t1;
        const not2Result = 1.0 - t2;

        // Update progress bar widths
        fillAnd.style.width = `${andResult * 100}%`;
        fillOr.style.width = `${orResult * 100}%`;
        fillNot1.style.width = `${not1Result * 100}%`;
        fillNot2.style.width = `${not2Result * 100}%`;

        // Update value text
        valAnd.textContent = andResult.toFixed(4);
        valOr.textContent = orResult.toFixed(4);
        valNot1.textContent = not1Result.toFixed(4);
        valNot2.textContent = not2Result.toFixed(4);
    }

    sliderT1.addEventListener('input', updateCalculator);
    sliderT2.addEventListener('input', updateCalculator);
    
    // Initial compute
    updateCalculator();
}

// Routing Softmax Simulator with Temperature and Cutoff
function setupRoutingSimulator() {
    const logitAnd = document.getElementById('logit-and');
    const logitOr = document.getElementById('logit-or');
    const logitNot = document.getElementById('logit-not');
    const sliderTemp = document.getElementById('slider-temp');
    const sliderCutoff = document.getElementById('slider-cutoff');

    const valLogitAnd = document.getElementById('val-logit-and');
    const valLogitOr = document.getElementById('val-logit-or');
    const valLogitNot = document.getElementById('val-logit-not');
    const valTemp = document.getElementById('val-temp');
    const valCutoff = document.getElementById('val-cutoff');

    const barAnd = document.getElementById('bar-and');
    const barOr = document.getElementById('bar-or');
    const barNot = document.getElementById('bar-not');

    const pctAnd = document.getElementById('pct-and');
    const pctOr = document.getElementById('pct-or');
    const pctNot = document.getElementById('pct-not');

    const labelAnd = document.getElementById('lbl-and');
    const labelOr = document.getElementById('lbl-or');
    const labelNot = document.getElementById('lbl-not');

    function updateSimulator() {
        const sAnd = parseFloat(logitAnd.value);
        const sOr = parseFloat(logitOr.value);
        const sNot = parseFloat(logitNot.value);
        const temp = parseFloat(sliderTemp.value);
        const cutoff = parseFloat(sliderCutoff.value);

        // Update value bubbles
        valLogitAnd.textContent = sAnd.toFixed(1);
        valLogitOr.textContent = sOr.toFixed(1);
        valLogitNot.textContent = sNot.toFixed(1);
        valTemp.textContent = temp.toFixed(2);
        valCutoff.textContent = cutoff.toFixed(2);

        // Step 1: Divide by temperature
        const zAnd = sAnd / temp;
        const zOr = sOr / temp;
        const zNot = sNot / temp;

        // Step 2: Softmax weights
        const maxLogit = Math.max(zAnd, zOr, zNot); // Subtract max for numerical stability
        const expAnd = Math.exp(zAnd - maxLogit);
        const expOr = Math.exp(zOr - maxLogit);
        const expNot = Math.exp(zNot - maxLogit);
        const sumExp = expAnd + expOr + expNot;

        let wAnd = expAnd / sumExp;
        let wOr = expOr / sumExp;
        let wNot = expNot / sumExp;

        // Save original weights for argmax fallback determination
        const origWeights = [wAnd, wOr, wNot];
        const maxIdx = origWeights.indexOf(Math.max(...origWeights));

        // Step 3: Apply Cutoff Threshold (matching PyTorch _apply_cutoff logic)
        let masked = [
            wAnd >= cutoff ? wAnd : 0,
            wOr >= cutoff ? wOr : 0,
            wNot >= cutoff ? wNot : 0
        ];
        
        const sumMasked = masked[0] + masked[1] + masked[2];

        if (sumMasked > 0) {
            // Renormalize
            wAnd = masked[0] / sumMasked;
            wOr = masked[1] / sumMasked;
            wNot = masked[2] / sumMasked;
        } else {
            // Fallback to one-hot argmax of original weights
            wAnd = (maxIdx === 0) ? 1.0 : 0.0;
            wOr = (maxIdx === 1) ? 1.0 : 0.0;
            wNot = (maxIdx === 2) ? 1.0 : 0.0;
        }

        // Update UI
        barAnd.style.width = `${wAnd * 100}%`;
        barOr.style.width = `${wOr * 100}%`;
        barNot.style.width = `${wNot * 100}%`;

        pctAnd.textContent = `${(wAnd * 100).toFixed(1)}%`;
        pctOr.textContent = `${(wOr * 100).toFixed(1)}%`;
        pctNot.textContent = `${(wNot * 100).toFixed(1)}%`;

        // Highlight active channels (opacity or color change)
        toggleActiveStyle(labelAnd, wAnd > 0);
        toggleActiveStyle(labelOr, wOr > 0);
        toggleActiveStyle(labelNot, wNot > 0);
    }

    function toggleActiveStyle(element, isActive) {
        if (isActive) {
            element.style.color = 'var(--text-primary)';
            element.style.opacity = '1';
        } else {
            element.style.color = 'var(--text-muted)';
            element.style.opacity = '0.5';
        }
    }

    logitAnd.addEventListener('input', updateSimulator);
    logitOr.addEventListener('input', updateSimulator);
    logitNot.addEventListener('input', updateSimulator);
    sliderTemp.addEventListener('input', updateSimulator);
    sliderCutoff.addEventListener('input', updateSimulator);

    // Initial simulation compute
    updateSimulator();
}

// Clipboard copying for BibTeX citation
function setupCitationCopy() {
    const copyBtn = document.getElementById('copy-citation-btn');
    const citationCode = document.getElementById('citation-code').textContent;

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(citationCode).then(() => {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                copyBtn.style.background = 'var(--accent-success)';
                copyBtn.style.color = 'white';

                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.style.background = '';
                    copyBtn.style.color = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy: ', err);
            });
        });
    }
}
