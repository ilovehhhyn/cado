const outcomes = {
  clean: ['Generate a rollout', 'Reward recorded: 1.0', 'Clean outcome', 'failure = None', 'Keep the reward', 'A valid signal for the trainer.'],
  infra: ['Generate a rollout', 'CUDA out of memory', 'Infrastructure failure', 'failure.failure_class = "infra"', 'Mask the unit', 'Keep this failure out of the reward.'],
  grader: ['Grade a rollout', 'No reward returned', 'Grader failure', 'failure.kind = "missing_reward"', 'Mask the unit', 'A missing reward is not a zero.']
};
const fields = ['flow-work', 'flow-event', 'flow-class', 'flow-code', 'flow-decision', 'flow-result'];
document.querySelectorAll('[data-outcome]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-outcome]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  outcomes[button.dataset.outcome].forEach((text, i) => { document.getElementById(fields[i]).textContent = text; });
}));
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
