'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const formatTime = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const tasks = {
  1: { title: 'Cover blocks', date: '2026.09.24', taskCount: 6, actionCount: 4,
    instruction: 'Cover the red, green, and blue blocks in that order, using both arms sequentially. The recorded instruction does not assign a particular cup to each block.',
    knowledge: 'Simulation-derived cover knowledge is supplied to the planner through the earlier ICL implementation. This run is later incorporated as a source of real-world experience.',
    outcome: 'Automatically verified success. All three blocks are covered in the required order.',
    phases: [] },
  2: { title: 'Press by number', date: '2026.10.01', taskCount: 12, actionCount: 13,
    instruction: 'Read the two numbers. Press green twice for the left number, blue once for the right number, then red once to confirm. Use a closed empty gripper for pressing.',
    knowledge: 'Simulation priors plus maintained Task 1 experience form the input store. Task retrieval supports finishing the current count before moving to the next target.',
    outcome: 'Human-confirmed success. Individual press effects were verified; the automatic completion chain did not accept all events, and the run was stopped.',
    phases: [
      'Task retrieval supports the first green press toward the left displayed count of two. The blue count follows both verified green presses.',
      'The first count entry is still in progress. One additional green press completes the left displayed count before moving to blue.',
      'Task retrieval selects the right displayed count: one blue press. Confirmation follows completion of both count entries.',
      'Both counts are complete. The selected Task Knowledge supports the final red confirmation press.'
    ] },
  3: { title: 'Uncover, count & press', date: '2026.10.01', taskCount: 16, actionCount: 16,
    instruction: 'Lift and set aside the cup, count the exposed blocks, then use the other arm to press the matching buttons in red–green–blue order.',
    knowledge: 'The input store combines simulation knowledge with maintained experience from Tasks 1 and 2. The recorded counts are red ×2, green ×1, blue ×1.',
    outcome: 'Human-confirmed success. Automatic verification of the blue-button effect remains unresolved; an additional stroke was rejected.',
    phases: [
      'The input store contains 16 Task and 16 Action entries. Grasping the cup is a prerequisite to revealing and counting the blocks.',
      'Lift the cup clear while preserving the grasp. Fresh observation reveals the blocks; exposing them does not yet complete safe cup placement.',
      'Action entry 6 supports a clear, supported placement approached from above. Adoption is recorded; behavior_changed=false. This entry also exists in simulation knowledge.',
      'Release and withdraw before transferring work to the other arm. The cup must stay stable and the button workspace must remain clear.',
      'The right arm prepares for pressing while the left arm stays clear. This excerpt shows preparation, not a completed press.',
      'Task entry 8 supports one distinct depression, release, and verification before the second red press. It carries maintenance provenance from Task 2.',
      'Task entry 9 supports the second distinct red press-and-release cycle, with verification before advancing to green. Task-level behavior change is recorded.',
      'The right arm approaches the green target. The run subsequently refreshes its visual identity and contact geometry before pressing.',
      'The robot rebinds the green cap despite a stale class label, then executes a press-and-release cycle. Its physical effect is verified.',
      'The blue contact stage remains uncertain in the automatic record. The later additional stroke request was rejected; no verified blue actuation is claimed.',
      'The robot withdraws for verification. Blue remains unconfirmed automatically; the operator separately confirms the overall task outcome.'
    ] }
};
let selectedTask = 3;
let currentChapter = -1;
const demoVideo = $('#demo-video');
const mediaFor = () => (window.MEDIA_DATA || []).find(item => item.task === selectedTask);

function renderTask(id, userInitiated = false) {
  const wasPlaying = !demoVideo.paused;
  demoVideo.pause();
  selectedTask = id;
  const task = tasks[id];
  $$('.task-tabs button').forEach(button => {
    const selected = Number(button.dataset.task) === id;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $('#demo-panel').setAttribute('aria-labelledby', `task-tab-${id}`);
  $('#demo-title').textContent = task.title;
  $('#demo-run').textContent = `ROLLOUT / ${task.date}`;
  $('#demo-instruction').textContent = task.instruction;
  $('#task-count').textContent = task.taskCount;
  $('#action-count').textContent = task.actionCount;
  $('#demo-knowledge').textContent = task.knowledge;
  $('#demo-outcome').textContent = task.outcome;
  $('#download-video').href = `assets/videos/task-${id}.mp4`;
  demoVideo.poster = `assets/images/task-${id}.jpg`;
  demoVideo.setAttribute('aria-label', `${task.title}, edited real robot demonstration at three times recorded speed`);
  if (userInitiated) {
    demoVideo.src = `assets/videos/task-${id}.mp4`;
    demoVideo.load();
  }
  $('#chapters').replaceChildren();
  const media = mediaFor();
  if (media) media.segments.forEach((segment, index) => {
    const button = document.createElement('button');
    const time = document.createElement('span');
    time.textContent = formatTime(segment.start);
    button.append(time, segment.label);
    button.setAttribute('aria-label', `Play ${segment.label}, excerpt ${formatTime(segment.start)}`);
    button.addEventListener('click', () => {
      demoVideo.currentTime = segment.start;
      updateChapter();
      demoVideo.play().catch(() => {});
    });
    $('#chapters').append(button);
  });
  currentChapter = -1;
  updateChapter(0);
  if (userInitiated && wasPlaying) demoVideo.play().catch(() => {});
}

function updateChapter(forcedTime) {
  const media = mediaFor();
  if (!media) return;
  const time = typeof forcedTime === 'number' ? forcedTime : demoVideo.currentTime;
  let index = media.segments.findIndex(segment => time >= segment.start && time < segment.end);
  if (index < 0) index = time >= media.duration ? media.segments.length - 1 : 0;
  const segment = media.segments[index];
  const source = Math.min(segment.sourceEnd, segment.sourceStart + (time - segment.start) * media.playbackSpeed);
  $('#source-time').textContent = `SOURCE ${formatTime(Math.max(0, source))}`;
  if (index !== currentChapter) {
    currentChapter = index;
    $('#current-chapter').textContent = segment.label;
    $$('#chapters button').forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
    $('#demo-knowledge').textContent = tasks[selectedTask].phases[index] || tasks[selectedTask].knowledge;
  }
}
demoVideo.addEventListener('timeupdate', () => updateChapter());
demoVideo.addEventListener('loadedmetadata', () => updateChapter());
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) demoVideo.pause();
}, {threshold: .05}).observe(demoVideo);
$$('.task-tabs button').forEach((button, index, buttons) => {
  button.addEventListener('click', () => renderTask(Number(button.dataset.task), true));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
    if (event.key === 'ArrowLeft') next = (index + buttons.length - 1) % buttons.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = buttons.length - 1;
    if (next === undefined) return;
    event.preventDefault(); buttons[next].focus(); buttons[next].click();
  });
});
renderTask(3);

const heroVideo = $('#hero-video');
const heroButton = $('#hero-toggle');
let heroUserPaused = false;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
function updateHeroButton() {
  heroButton.setAttribute('aria-label', heroVideo.paused ? 'Play overview video' : 'Pause overview video');
  heroButton.firstElementChild.textContent = heroVideo.paused ? '▶' : 'Ⅱ';
  $('.hero-toggle-label').textContent = heroVideo.paused ? 'Play video' : 'Pause video';
}
heroButton.addEventListener('click', () => {
  if (heroVideo.paused) { heroUserPaused = false; heroVideo.play().catch(() => {}); }
  else { heroUserPaused = true; heroVideo.pause(); }
});
heroVideo.addEventListener('play', updateHeroButton);
heroVideo.addEventListener('pause', updateHeroButton);
if (reducedMotion.matches) { heroVideo.autoplay = false; heroUserPaused = true; heroVideo.pause(); }
new IntersectionObserver(entries => {
  if (!entries[0].isIntersecting) heroVideo.pause();
  else if (!heroUserPaused && !reducedMotion.matches) heroVideo.play().catch(() => {});
}, {threshold: .1}).observe(heroVideo);
document.addEventListener('visibilitychange', () => { if (document.hidden) {heroVideo.pause();demoVideo.pause();} });
updateHeroButton();

const benchmarkTasks = ['Rearrange Blocks', 'Swap Blocks', 'Press Button', 'Swap T', 'Put Back Block', 'Cover Blocks'];
const benchmarks = {
  'Qwen3.8-27B': {base:[15,10,35,20,20,15], full:[35,25,60,40,50,40], mean:[19.2,41.7], gain:22.5},
  'GPT-5.5': {base:[50,45,95,55,50,55], full:[80,70,100,90,75,80], mean:[58.3,82.5], gain:24.2},
  'GPT-6': {base:[75,70,100,45,75,70], full:[90,85,100,95,90,90], mean:[72.5,91.7], gain:19.2}
};
function renderBenchmark(model) {
  const data = benchmarks[model];
  $$('[data-model]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.model === model)));
  $('#model-label').textContent = `${model} · SIX-TASK MEAN`;
  $('#mean-base').textContent = data.mean[0].toFixed(1);
  $('#mean-full').innerHTML = `${data.mean[1].toFixed(1)}<span>%</span>`;
  $('#mean-gain').textContent = `+${data.gain.toFixed(1)} pp`;
  $('#table-model').textContent = model;
  $('#benchmark-bars').innerHTML = benchmarkTasks.map((task, i) => `<div class="bar-row" aria-label="${task}: without HPK ${data.base[i]} percent; Full HPK ${data.full[i]} percent"><span>${task}</span><div class="bar-pair" aria-hidden="true"><div class="bar" style="width:${data.base[i]}%"><span>${data.base[i]}</span></div><div class="bar full" style="width:${data.full[i]}%"><span>${data.full[i]}</span></div></div></div>`).join('');
  $('#benchmark-table').innerHTML = benchmarkTasks.map((task, i) => `<tr><th scope="row">${task}</th><td>${data.base[i]}</td><td>${data.full[i]}</td><td>+${data.full[i]-data.base[i]}</td></tr>`).join('') + `<tr><th scope="row">Overall</th><td>${data.mean[0]}</td><td>${data.mean[1]}</td><td>+${data.gain}</td></tr>`;
}
$$('[data-model]').forEach(button => button.addEventListener('click', () => renderBenchmark(button.dataset.model)));
renderBenchmark('GPT-5.5');

function renderLearning() {
  const data = window.LEARNING_DATA;
  if (!data) return;
  const x = k => 47 + k / 80 * 387;
  const y = percent => 231 - (percent - 40) / 60 * 192;
  let svg = '<svg viewBox="0 0 490 280" role="img" aria-labelledby="learning-title learning-desc"><title id="learning-title">Held-out success improves with interaction</title><desc id="learning-desc">Across 0 to 80 learning rollouts, GPT-5.5 improves from 48.3 to 75.0 percent, and GPT-6 from 70.0 to 88.3 percent. Error bars are sample standard deviations over three learning histories.</desc>';
  svg += '<text x="47" y="18" font-size="10" fill="#61706b" font-family="Arial">HELD-OUT SUCCESS (%)</text>';
  [40,60,80,100].forEach(tick => { svg += `<line x1="47" x2="445" y1="${y(tick)}" y2="${y(tick)}" stroke="#d5dfd4" stroke-dasharray="3 4"/><text x="33" y="${y(tick)+4}" text-anchor="end" font-size="11" fill="#61706b" font-family="Arial">${tick}</text>`; });
  [0,10,20,40,80].forEach(tick => { svg += `<text x="${x(tick)}" y="251" text-anchor="middle" font-size="11" fill="#61706b" font-family="Arial">${tick}</text>`; });
  svg += '<text x="244" y="275" text-anchor="middle" font-size="11" fill="#61706b" font-family="Arial">Learning rollouts, K</text>';
  [['GPT-5.5','#20594f'],['GPT-6','#a14d64']].forEach(([model,color]) => {
    const points = data[model];
    svg += `<polyline points="${points.map(p => `${x(p.k)},${y(p.mean)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
    points.forEach(p => { const px=x(p.k),top=y(p.mean+p.sd),bottom=y(p.mean-p.sd); svg += `<g><title>${model}, K=${p.k}: ${p.mean.toFixed(1)}% ± ${p.sd.toFixed(1)}</title><path d="M${px},${top}V${bottom}M${px-4},${top}h8M${px-4},${bottom}h8" stroke="${color}" opacity=".5"/><circle cx="${px}" cy="${y(p.mean)}" r="4" fill="${color}" stroke="#f0f3ef" stroke-width="1.5"/></g>`; });
    const last=points[points.length-1]; svg += `<text x="${x(last.k)-4}" y="${y(last.mean)-15}" text-anchor="end" font-size="14" font-weight="600" fill="${color}" font-family="Arial">${last.mean.toFixed(1)}%</text>`;
  });
  $('#learning-chart').innerHTML = svg + '</svg>';
  $('#learning-table').innerHTML = data['GPT-5.5'].map((p,i) => `<tr><th scope="row">${p.k}</th><td>${p.mean.toFixed(1)} ± ${p.sd.toFixed(1)}</td><td>${data['GPT-6'][i].mean.toFixed(1)} ± ${data['GPT-6'][i].sd.toFixed(1)}</td></tr>`).join('');
}
renderLearning();

const figureDialog = $('#figure-dialog');
$$('[data-figure]').forEach(button => button.addEventListener('click', () => {
  $('#dialog-image').src = button.dataset.figure;
  $('#dialog-image').alt = button.dataset.title;
  $('#figure-title').textContent = button.dataset.title;
  figureDialog.showModal();
}));
$('#close-figure').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('click', event => {if (event.target === figureDialog) {const rect=figureDialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)figureDialog.close();}});
$('#copy-citation').addEventListener('click', async () => {
  const citation = $('#bibtex').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    $('#copy-citation').textContent = 'Copied';
    $('#copy-status').textContent = 'Citation copied to clipboard.';
    setTimeout(() => {$('#copy-citation').textContent = 'Copy citation';}, 2000);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();range.selectNodeContents($('#bibtex'));selection.removeAllRanges();selection.addRange(range);
    $('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    $('#copy-citation').textContent = 'Selected — press Ctrl/Cmd+C';
  }
});
const sectionObserver = new IntersectionObserver(entries => entries.forEach(entry => {
  if (!entry.isIntersecting) return;
  $$('.site-header nav a').forEach(link => {
    if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current','location');
    else link.removeAttribute('aria-current');
  });
}), {rootMargin:'-15% 0px -55% 0px'});
['top','overview','demos','method','results','citation'].forEach(id => sectionObserver.observe(document.getElementById(id)));
