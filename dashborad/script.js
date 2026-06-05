function updateClock() {

  const now = new Date();

  let hours = now.getHours();
  const minutes = String(now.getMinutes()).padStart(2,'0');
  const seconds = String(now.getSeconds()).padStart(2,'0');

  const ampm = hours >= 12 ? 'PM' : 'AM';

  hours = hours % 12;
  hours = hours ? hours : 12;

  document.getElementById('time').innerHTML =
    `${hours}:${minutes}:${seconds} ${ampm}`;

  document.getElementById('date').innerHTML =
    now.toDateString();
}

setInterval(updateClock,1000);

updateClock();