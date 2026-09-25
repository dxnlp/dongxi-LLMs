/* Local GIF/static controls. No requests, tracking, or cloud-editor actions. */
(() => {
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const setPlaying = (figure, playing) => {
    const image = figure.querySelector('img');
    const button = figure.querySelector('button');
    image.src = playing ? image.dataset.gif : image.dataset.still;
    button.setAttribute('aria-pressed', String(playing));
    button.textContent = playing ? '显示静态图' : '播放动画';
  };
  const figures = [...document.querySelectorAll('figure.motion')];
  for (const figure of figures) {
    const button = figure.querySelector('button');
    button.hidden = false;
    setPlaying(figure, !preference.matches);
    button.addEventListener('click', () => {
      setPlaying(figure, button.getAttribute('aria-pressed') !== 'true');
    });
  }
  preference.addEventListener('change', () => {
    for (const figure of figures) setPlaying(figure, !preference.matches);
  });
})();
