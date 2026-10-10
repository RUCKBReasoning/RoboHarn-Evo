'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const formatTime = seconds => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
const tasks = {
  1: {title: 'Cover blocks', instruction: 'Cover red, then green, then blue. Use both arms, moving only one arm at a time.'},
  2: {title: 'Press by digit–color mapping', instruction: 'Read the left and right digits. In this demonstration, the left digit sets the green-button count and the right digit sets the blue-button count. Complete green, then blue, then press red once to finish.'},
  3: {title: 'Uncover, count & press', instruction: 'Use one arm to uncover the blocks and release the cup safely away from them. Count each color, then use the other arm to press red, green, and blue in order, matching each color’s block count. No extra confirmation press.'}
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
  $('#demo-instruction').textContent = task.instruction;
  $('#demo-outcome').textContent = 'Task success verified.';
  $('#download-video').href = mediaFor().video;
  demoVideo.poster = `assets/images/task-${id}.jpg`;
  demoVideo.setAttribute('aria-label', `${task.title}, continuous real robot demonstration`);
  if (userInitiated) {
    demoVideo.src = mediaFor().video;
    demoVideo.load();
  }
  $('#chapters').replaceChildren();
  const media = mediaFor();
  if (media) media.segments.forEach((segment, index) => {
    const button = document.createElement('button');
    const time = document.createElement('span');
    time.textContent = formatTime(segment.start);
    button.append(time, segment.label);
    button.setAttribute('aria-label', `Play ${segment.label}, ${formatTime(segment.start)}`);
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
  $('#source-time').textContent = `${formatTime(time)} / ${formatTime(media.duration)}`;
  $('#reasoning-progress').style.width = `${Math.max(0, Math.min(100, (time-segment.start)/(segment.end-segment.start)*100))}%`;
  if (index !== currentChapter) {
    currentChapter = index;
    $('#current-chapter').textContent = segment.label;
    $$('#chapters button').forEach((button, i) => button.setAttribute('aria-current', String(i === index)));
    const reasoning = window.REASONING_DATA[selectedTask][index];
    $('#reasoning-step').textContent = `${String(index+1).padStart(2,'0')} / ${String(media.segments.length).padStart(2,'0')}`;
    $('#reasoning-title').textContent = reasoning.title;
    ['plan','action','check'].forEach(key => $(`#reasoning-${key}`).textContent = reasoning[key]);
    $('#reasoning-content').getAnimations().forEach(animation => animation.cancel());
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) $('#reasoning-content').animate([{opacity:.35,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:220});
  }
}
demoVideo.addEventListener('timeupdate', () => updateChapter());
demoVideo.addEventListener('loadedmetadata', () => updateChapter());
demoVideo.addEventListener('seeked', () => updateChapter());
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
heroVideo.addEventListener('timeupdate', () => {
  const segment = window.HERO_DATA?.segments.find(item => heroVideo.currentTime >= item.start && heroVideo.currentTime < item.end);
  if (segment) $('#hero-speed').textContent = segment.label.toUpperCase();
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
['top','overview','method','results','story','demos','citation'].forEach(id => sectionObserver.observe(document.getElementById(id)));
