'use strict';
(() => {
  const film = document.getElementById('story-video');
  if (!film) return;
  const section = document.getElementById('story');
  const chapters = [...section.querySelectorAll('[data-story-time]')];
  let pendingSeek = null;
  function sync() {
    const time = film.currentTime;
    chapters.forEach((button, index) => {
      const next = chapters[index + 1];
      button.setAttribute('aria-current', String(time >= Number(button.dataset.storyTime) && (!next || time < Number(next.dataset.storyTime))));
    });
  }
  function seek(time) {
    // Land after the brief scene fade so chapter previews stay legible.
    time += .3;
    if (film.readyState === 0) {
      pendingSeek = time;
      film.load();
      return;
    }
    film.currentTime = time;
    sync();
  }
  film.addEventListener('loadedmetadata', () => {
    if (pendingSeek !== null) {
      film.currentTime = pendingSeek;
      pendingSeek = null;
      sync();
    }
  });
  chapters.forEach(button => button.addEventListener('click', () => {
    seek(Number(button.dataset.storyTime));
    film.play().catch(() => {});
  }));
  film.addEventListener('timeupdate', sync);
  film.addEventListener('seeked', sync);
  film.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(video => { if (video !== film) video.pause(); });
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) film.pause();
  }, {threshold: .08}).observe(film);
  sync();
})();
