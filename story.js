'use strict';
(() => {
  const film = document.getElementById('story-video');
  if (!film) return;
  film.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(video => {
      if (video !== film) video.pause();
    });
  });
  new IntersectionObserver(entries => {
    if (!entries[0].isIntersecting) film.pause();
  }, {threshold: .08}).observe(film);
})();
