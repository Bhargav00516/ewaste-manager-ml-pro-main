function rotateAndRedirect(url) {
  const cards = document.querySelectorAll('.icon-card, .help-card');
  cards.forEach(card => {
    if (card.onclick.toString().includes(url)) {
      card.querySelector('img').style.transition = 'transform 0.3s ease';
      card.querySelector('img').style.transform = 'rotate(360deg)';
      setTimeout(() => {
        window.location.href = url;
      }, 300);
    }
  });
}
