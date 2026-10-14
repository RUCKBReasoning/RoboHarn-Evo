'use strict';
(() => {
  const section = document.getElementById('demos');
  const data = window.DEMO_SETTINGS, evidence = window.DEMO_EVIDENCE;
  if (!section || !data || !evidence) return;
  const cards = [...section.querySelectorAll('[data-setting]')];
  const tabs = [...section.querySelectorAll('[data-demo-task]')];
  const pairButton = document.getElementById('play-settings');
  const milestones = document.getElementById('decision-milestones');
  const videos = cards.map(card => card.querySelector('video'));
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let task = '3', generation = 0, pendingJump = 0;
  const setText = (card, field, value) => { const e = card.querySelector(`[data-${field}]`); if(e && e.textContent !== value) e.textContent = value; };
  function syncPairButton() {
    pairButton.textContent = videos.some(video => !video.paused) ? 'Pause both' : 'Play both';
  }
  function syncMoment(card) {
    const setting = card.dataset.setting, item = data[setting][task], record = evidence.episodes[setting][task];
    const video = card.querySelector('video'), t = video.currentTime;
    const decision = [...record.decisions].reverse().find(s => t >= s.start) || record.decisions[0];
    setText(card, 'task-level', decision.taskLevel);
    setText(card, 'action-level', decision.actionLevel);
    let label='This episode', kind='Current task', rule=decision.context, image=item.poster;
    // Recompute from time so scrubbing backwards restores the matching source rule.
    const past = record.adoptions.filter(e => e.start <= t);
    const current = past.filter(e => e.start >= decision.start - .01);
    const adopted = current.find(e => e.layer === 'task') || current[current.length - 1];
    if (setting === 'rsi') {
      if (task === '1') {
        label='From simulation';kind='Prior experience';image='assets/images/sim-cover.jpg';
        rule='Cover with support, then release and verify.';

      } else {
        if(adopted){
          label=adopted.origin.includes('Maintained after Press')?'From Press by number':adopted.origin.includes('Cover simulation')?'From cover simulation':'From press simulation';kind='Earlier experience';image=adopted.image;rule=adopted.rule;
        }

      }
    }
    setText(card,'origin-label',label);setText(card,'rule',rule);
    const img=card.querySelector('[data-origin-image]');if(img.getAttribute('src')!==image)img.src=image;
    img.alt=kind+': '+(adopted ? label : item.title);
    const origin=card.querySelector('[data-origin-card]');
    origin.classList.toggle('is-memory',setting==='rsi'&&(!!adopted||task==='1'));
    const key=`${task}:${decision.traceIndex}:${adopted ? adopted.sourceLine : 'observation'}`;
    if (card.dataset.evidenceKey !== key || card.dataset.replayEvidence) {
      const changed = !!card.dataset.evidenceKey;
      delete card.dataset.replayEvidence;
      card.dataset.evidenceKey=key;
      card.querySelectorAll('.experience-origin,.decision-connector circle,.current-decision').forEach(el=>el.getAnimations().forEach(a=>a.cancel()));
      if(changed && !reduceMotion.matches) {
        origin.animate([{opacity:.4,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:520,easing:'ease-out'});
        card.querySelector('.decision-connector circle').animate([{transform:'translateY(0)',opacity:0},{opacity:1,offset:.3},{transform:'translateY(20px)',opacity:0}],{duration:1150,easing:'ease-in-out'});
        card.querySelector('.current-decision').animate([{borderColor:'#b7c6ba',boxShadow:'0 0 18px #d0dfb824'},{borderColor:'#607568',boxShadow:'0 0 0 #d0dfb800'}],{duration:1400});
      }
    }
  }
  function jumpTo(id) {
    const moment=evidence.tasks[task].moments.find(m=>m.id===id);if(!moment)return;
    const token=++pendingJump, version=generation;
    videos.forEach(v=>v.pause());
    milestones.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.momentId===id)));
    cards.forEach(card=>{
      const video=card.querySelector('video');
      const seek=()=>{
        if(version!==generation||token!==pendingJump)return;
        card.dataset.replayEvidence = 'true';
        video.currentTime=Math.min(moment[card.dataset.setting],Math.max(0,video.duration-.1));
        syncMoment(card);
      };
      if(video.readyState>=1)seek();else video.addEventListener('loadedmetadata',seek,{once:true});
    });
    document.getElementById('decision-jump-status').textContent=`Both recordings positioned at: ${moment.label}. Press Play both to continue.`;
    syncPairButton();
  }
  function chooseTask(id, momentId) {
    videos.forEach(video=>video.pause());generation++;task=id;
    setText(section,'instruction',data.tasks[task].instruction);
    tabs.forEach(tab=>{const selected=tab.dataset.demoTask===task;tab.setAttribute('aria-selected',String(selected));tab.tabIndex=selected?0:-1;});
    document.getElementById('setting-panels').setAttribute('aria-labelledby',`setting-task-${task}`);
    milestones.replaceChildren();
    evidence.tasks[task].moments.forEach(m=>{
      const b=document.createElement('button');b.type='button';b.dataset.momentId=m.id;b.textContent=m.label;b.setAttribute('aria-pressed','false');
      b.addEventListener('click',()=>jumpTo(m.id));milestones.append(b);
    });
    cards.forEach(card=>{
      delete card.dataset.evidenceKey;
      const item=data[card.dataset.setting][task], video=card.querySelector('video');
      card.querySelectorAll('[data-flow-task]').forEach(node=>node.classList.toggle('selected',node.dataset.flowTask===task));
      video.poster=item.poster;video.setAttribute('aria-label',`${card.querySelector('h3').textContent}: ${item.title}`);
      video.src=item.video;video.load();syncMoment(card);
    });
    const valid = evidence.tasks[task].moments.some(m=>m.id===momentId);
    jumpTo(valid ? momentId : evidence.tasks[task].defaultMoment);
  }
  tabs.forEach((tab,index)=>{
    tab.addEventListener('click',()=>chooseTask(tab.dataset.demoTask));
    tab.addEventListener('keydown',event=>{
      const next={ArrowRight:(index+1)%tabs.length,ArrowLeft:(index+tabs.length-1)%tabs.length,Home:0,End:tabs.length-1}[event.key];
      if(next===undefined)return;event.preventDefault();tabs[next].focus();tabs[next].click();
    });
  });
  pairButton.addEventListener('click',()=>{
    if(videos.some(v=>!v.paused))videos.forEach(v=>v.pause());
    else {if(videos.every(v=>v.ended))jumpTo('start');videos.forEach(v=>v.play().catch(()=>{}));}
  });
  cards.forEach(card=>{
    const video=card.querySelector('video');
    ['timeupdate','seeked','loadedmetadata'].forEach(e=>video.addEventListener(e,()=>syncMoment(card)));
    ['play','pause','ended'].forEach(e=>video.addEventListener(e,syncPairButton));
    video.addEventListener('play',()=>document.querySelectorAll('video').forEach(v=>{if(!videos.includes(v))v.pause();}));
    video.addEventListener('error',()=>{document.getElementById('decision-jump-status').textContent='Video could not load. Select another task or reload to retry.';});
  });
  reduceMotion.addEventListener('change',()=>{if(reduceMotion.matches)section.getAnimations({subtree:true}).forEach(a=>a.cancel());});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)videos.forEach(v=>v.pause());}).observe(section);
  const params=new URLSearchParams(location.search), requestedTask=params.get('task');
  chooseTask(['1','2','3'].includes(requestedTask)?requestedTask:'3',params.get('moment'));
})();
