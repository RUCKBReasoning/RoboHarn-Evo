'use strict';
(() => {
  const film = document.getElementById('story-video');
  if (!film) return;
  const section = document.getElementById('story');
  const nodes = [...section.querySelectorAll('[data-memory]')];
  const chapters = [...section.querySelectorAll('[data-story-time]')];
  const proofButtons = [...section.querySelectorAll('[data-evidence]')];
  const stages = {
    sim: {time:0, kicker:'SIMULATION / STARTING KNOWLEDGE',title:'Experience has a starting point.',description:'The cover and press simulation libraries contain 9 Task and 7 Action entries in total. Task 1 uses only the cover library (6 Task / 4 Action) through the earlier planner-only ICL implementation.',provenance:'The combined simulation pool and Task 1 experience are maintained into the input store for Task 2.'},
    cover: {time:12,kicker:'TASK 1 / PHYSICAL EXPERIENCE',title:'A trajectory becomes evidence.',description:'The robot covers the blocks in order. The recorded trajectory and physical feedback are incorporated into knowledge maintenance. The resulting Task 2 input contains 12 Task and 13 Action entries.',provenance:'The robot completes the covering task. Its experience is used to update the knowledge available to Task 2.'},
    press: {time:36,kicker:'TASK 2 / KNOWLEDGE MAINTENANCE',title:'A counting strategy carries forward.',description:'The robot enters two counts, then confirms. Task and Action Knowledge support its decisions. Maintenance with this trajectory produces the 16 Task / 16 Action input store subsequently used by Task 3.',provenance:'The pressing strategies carried forward from this task are later retrieved for the two red presses in Task 3.'},
    compose: {time:58,kicker:'TASK 3 / CROSS-TASK RETRIEVAL',title:'Retrieve for the current goal.',description:'Uncover the blocks, set the cup aside, count, and press. Task Knowledge from the preceding count-entry task is retrieved for two red presses in this new task. Action Knowledge supports placement and contact geometry.',provenance:'Select an example to explore the pressing and placement knowledge retrieved in this task.'}
  };
  const evidence = {
    task8:{time:71, layer:'TASK KNOWLEDGE / ENTRY #8',title:'One press. Release. Verify. Then continue.',description:'For the first of two red presses, the retrieved strategy refines the current subtask: verify a distinct depression, release, and withdrawal before the second press.',provenance:'A pressing strategy carried forward from Task 2 and reused for the first red press in Task 3.',text:'Task 2 → Task 3: check that the first press is complete before starting the second.'},
    task9:{time:87,layer:'TASK KNOWLEDGE / ENTRY #9',title:'Complete the second cycle before advancing.',description:'The next retrieved strategy preserves the first verified red press, completes a distinct second press-and-release cycle, and waits for confirmation before moving to green.',provenance:'Task 3 reuses counting knowledge carried forward from Task 2 to finish both red presses in order.',text:'Finish both red presses before moving to green, so each required press is counted separately.'},
    action6:{time:62,layer:'ACTION KNOWLEDGE / ENTRY #6',title:'Set the cup aside with support and clearance.',description:'The retrieved knowledge describes how to set the cup down: choose a supported location, approach from above, and leave enough clearance.',provenance:'This placement knowledge comes from simulation and remains available after Tasks 1 and 2.',text:'Task 3 retrieves an existing placement strategy when setting the cup aside.'}
  };
  let currentKey = '';
  const setText = (id, value) => document.getElementById(id).textContent = value;
  function showStage(key) {
    const s = stages[key];
    if (currentKey === key) return;
    currentKey = key;
    nodes.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.memory===key)));
    proofButtons.forEach(b=>b.setAttribute('aria-pressed','false'));
    setText('memory-kicker',s.kicker);setText('memory-title',s.title);setText('memory-description',s.description);setText('memory-provenance',s.provenance);
    setText('memory-proof-text','Select an example to see how earlier knowledge helps with the current task.');
  }
  function showEvidence(key) {
    if(currentKey===key)return;
    currentKey=key;const e=evidence[key];
    nodes.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.memory==='compose')));
    proofButtons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.evidence===key)));
    setText('memory-kicker',e.layer);setText('memory-title',e.title);setText('memory-description',e.description);setText('memory-provenance',e.provenance);setText('memory-proof-text',e.text);
  }
  function sync() {
    const t=film.currentTime;
    chapters.forEach((b,i)=>b.setAttribute('aria-current',String(t>=Number(b.dataset.storyTime)&&(!chapters[i+1]||t<Number(chapters[i+1].dataset.storyTime)))));
    if(t>=87&&t<93)showEvidence('task9');else if(t>=71&&t<87)showEvidence('task8');else if(t>=62&&t<68)showEvidence('action6');else showStage(t>=58?'compose':t>=36?'press':t>=12?'cover':'sim');
  }
  let pendingSeek = null;
  function seek(time) {
    // Land after the brief scene fade so paused chapter previews stay legible.
    time += .3;
    if (film.readyState === 0) {
      pendingSeek = time;
      film.load();
      const key = time>=58?'compose':time>=36?'press':time>=12?'cover':'sim';
      showStage(key);
      return;
    }
    film.currentTime=time;sync();
  }
  film.addEventListener('loadedmetadata',()=>{
    if(pendingSeek!==null){film.currentTime=pendingSeek;pendingSeek=null;sync();}
  });
  chapters.forEach(b=>b.addEventListener('click',()=>{seek(Number(b.dataset.storyTime));film.play().catch(()=>{});}));
  nodes.forEach(b=>b.addEventListener('click',()=>seek(stages[b.dataset.memory].time)));
  proofButtons.forEach(b=>b.addEventListener('click',()=>{seek(evidence[b.dataset.evidence].time);showEvidence(b.dataset.evidence);}));
  film.addEventListener('timeupdate',sync);film.addEventListener('seeked',sync);
  film.addEventListener('play',()=>{section.classList.add('story-is-playing');document.querySelectorAll('video').forEach(v=>{if(v!==film)v.pause();});});
  film.addEventListener('pause',()=>section.classList.remove('story-is-playing'));
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)film.pause();},{threshold:.08}).observe(film);
  sync();
})();
