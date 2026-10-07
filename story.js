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
    cover: {time:12,kicker:'TASK 1 / PHYSICAL EXPERIENCE',title:'A trajectory becomes evidence.',description:'The robot covers the blocks in order. The recorded trajectory and physical feedback are incorporated into knowledge maintenance. The resulting Task 2 input contains 12 Task and 13 Action entries.',provenance:'Task 1: 20260924/195945 · automatically verified success. Store counts are maintained snapshots, not new-entry totals.'},
    press: {time:36,kicker:'TASK 2 / KNOWLEDGE MAINTENANCE',title:'A counting strategy carries forward.',description:'The robot enters two counts, then confirms. Task and Action Knowledge support its decisions. Maintenance with this trajectory produces the 16 Task / 16 Action input store subsequently used by Task 3.',provenance:'Task 2: 20261001/133133 · human-confirmed success; automatic status interrupted. Task 3 entries #8 and #9 carry this episode’s maintenance provenance.'},
    compose: {time:58,kicker:'TASK 3 / CROSS-TASK RETRIEVAL',title:'Retrieve for the current goal.',description:'Uncover the blocks, set the cup aside, count, and press. Task Knowledge from the preceding count-entry task is retrieved for two red presses in this new task. Action Knowledge supports placement and contact geometry.',provenance:'Task 3: 20261001/210736 · 16 Task / 16 Action entries at input. Select a retrieval example to inspect the saved evidence.'}
  };
  const evidence = {
    task8:{time:71, layer:'TASK KNOWLEDGE / ENTRY #8',title:'One press. Release. Verify. Then continue.',description:'For the first of two red presses, the retrieved strategy refines the current subtask: verify a distinct depression, release, and withdrawal before the second press.',provenance:'Task 2 maintained checkpoint → Task 3 input, Task line 9 (zero-based index 8). Saved event 160; retrieval at approximately 26:33 in the raw Task 3 recording.',text:'A strategy maintained with Task 2 is adopted in Task 3. The saved log explicitly records a change to the task plan.',flags:[['Source','Task 2 maintenance'],['Adopted','Yes'],['Task behavior changed','Yes'],['Saved event','Task 3 · line 160']]},
    task9:{time:87,layer:'TASK KNOWLEDGE / ENTRY #9',title:'Complete the second cycle before advancing.',description:'The next retrieved strategy preserves the first verified red press, completes a distinct second press-and-release cycle, and waits for confirmation before moving to green.',provenance:'Task 2 maintained checkpoint → Task 3 input, Task line 10 (zero-based index 9). Saved event 201; retrieval at approximately 31:15 in the raw Task 3 recording.',text:'The second count strategy also carries Task 2 episode provenance and a recorded task-plan change.',flags:[['Source','Task 2 maintenance'],['Adopted','Yes'],['Task behavior changed','Yes'],['Saved event','Task 3 · line 201']]},
    action6:{time:62,layer:'ACTION KNOWLEDGE / ENTRY #6',title:'Set the cup aside with support and clearance.',description:'The selected placement knowledge favors a supported location, an approach from above, preserved attachment orientation, and clearance. The Action entry is adopted, but the saved log does not record a behavior change.',provenance:'Content matches the simulation cover library and the stores maintained with Tasks 1 and 2. It is not exclusively a new discovery from Task 1. Saved event 82.',text:'This example distinguishes knowledge adoption from behavior change, and shared provenance from a uniquely new rule.',flags:[['Source','Simulation + maintained stores'],['Adopted','Yes'],['Action behavior changed','No'],['Saved event','Task 3 · line 82']]}
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
    setText('memory-proof-text','Select an entry to see its source, its role in the current task, and the saved adoption result.');
    flags([['Source','Saved decision logs'],['Display','Editorial replay']]);
  }
  function flags(rows) {
    const list=document.getElementById('memory-proof-flags');list.replaceChildren();
    rows.forEach(([key,value])=>{const row=document.createElement('div'),term=document.createElement('dt'),detail=document.createElement('dd');term.textContent=key;detail.textContent=value;row.append(term,detail);list.append(row);});
  }
  function showEvidence(key) {
    if(currentKey===key)return;
    currentKey=key;const e=evidence[key];
    nodes.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.memory==='compose')));
    proofButtons.forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.evidence===key)));
    setText('memory-kicker',e.layer);setText('memory-title',e.title);setText('memory-description',e.description);setText('memory-provenance',e.provenance);setText('memory-proof-text',e.text);flags(e.flags);
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
