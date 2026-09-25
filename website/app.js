const scores = {easy: [87.38,80.66,80.66,86.55,92.00], hard: [73.73,70.75,69.53,68.90,94.15]};
const labels = ['tplane', 'RFT-FM', 'Anomaly Transformer', 'TranAD', 'Flag every run faulty'];
document.querySelectorAll('[data-difficulty]').forEach(button => button.addEventListener('click', () => {
  const split = button.dataset.difficulty;
  document.querySelectorAll('[data-difficulty]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  document.querySelectorAll('.chart-row').forEach((row, i) => {
    row.querySelector('.bar').style.width = `${scores[split][i]}%`;
    row.querySelector('.bar-value').textContent = scores[split][i].toFixed(2);
  });
  document.getElementById('split-label').textContent = split;
  const description = `${split} split F1: ` + labels.map((name, i) => `${name} ${scores[split][i].toFixed(2)}`).join(', ');
  document.querySelector('.chart').setAttribute('aria-label', description);
  document.getElementById('chart-announcement').textContent = description;
}));
document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const code = document.getElementById(button.dataset.copy);
  try {
    await navigator.clipboard.writeText(code.textContent);
    button.textContent = 'Copied ✓';
    document.getElementById('copy-status').textContent = 'Code copied to clipboard.';
    setTimeout(() => { button.textContent = 'Copy ↗'; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents(code);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    document.getElementById('copy-status').textContent = 'Select and copy the highlighted code manually.';
    button.textContent = 'Select & copy';
  }
}));

// A user-defined scenario, independent of the observed rollout failure statistics.
const savingsForm = document.getElementById('savings-form');
const savingsInputs = ['gpus', 'rate', 'late', 'early', 'recovery'].map(name => document.getElementById(`savings-${name}`));
const usd = new Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', maximumFractionDigits: 0});
const number = new Intl.NumberFormat('en-US', {maximumFractionDigits: 2});
function updateSavings() {
  const valid = savingsInputs.every(input => input.validity.valid && Number.isFinite(input.valueAsNumber));
  document.getElementById('savings-error').hidden = valid;
  savingsInputs.forEach(input => input.setAttribute('aria-invalid', String(!input.validity.valid)));
  const total = document.getElementById('savings-total');
  if (!valid) {
    total.textContent = '—';
    document.getElementById('savings-formula').textContent = 'Complete the inputs to calculate an estimate.';
    document.querySelector('.time-comparison').hidden = true;
    return;
  }
  const [gpus, rate, late, early, recovery] = savingsInputs.map(input => input.valueAsNumber);
  const hours = Math.max(0, late - early - recovery);
  const intervention = early + recovery;
  const scale = Math.max(late, intervention, 1);
  total.textContent = usd.format(gpus * rate * hours);
  document.getElementById('savings-formula').textContent = `${number.format(gpus)} GPUs × $${rate.toFixed(2)}/hour × ${number.format(hours)} hours of avoided waste`;
  document.querySelector('.time-comparison').hidden = false;
  document.getElementById('time-without').textContent = `${number.format(late)} h`;
  document.getElementById('time-with').textContent = `${number.format(intervention)} h`;
  document.getElementById('time-without-bar').style.width = `${late / scale * 100}%`;
  document.getElementById('time-with-bar').style.width = `${intervention / scale * 100}%`;
}
savingsForm.addEventListener('input', updateSavings);
savingsForm.addEventListener('submit', event => event.preventDefault());
updateSavings();

