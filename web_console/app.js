const consoleEl = document.getElementById('console');
const form = document.getElementById('promptForm');
const input = document.getElementById('promptInput');
const resetBtn = document.getElementById('resetBtn');
// Set to true to require server API responses (no local fallback)
const FORCE_SERVER = true;

function appendEntry(who, text, opts = {}){
  const div = document.createElement('div');
  div.className = `entry ${who}`;
  const whoSpan = document.createElement('span');
  whoSpan.className = 'who';
  whoSpan.textContent = who === 'user' ? 'You' : 'AI';
  const textSpan = document.createElement('span');
  textSpan.className = 'text';
  textSpan.innerHTML = text.replace(/\n/g, '<br>');
  div.appendChild(whoSpan);
  div.appendChild(textSpan);
  consoleEl.appendChild(div);
  consoleEl.scrollTop = consoleEl.scrollHeight;
  return div;
}

function simulateAI(prompt){
  // Basic simulated AI: friendly Romanian responses with small heuristics
  const p = prompt.trim();
  if(!p) return "Nu ai scris nimic—te rog încearcă din nou.";
  if(/^(help|ajutor|ce poți|ce faci)/i.test(p)){
    return "Sunt o consolă AI simulată. Trimite orice prompt, iar eu răspund aici.";
  }
  if(p.length < 30){
    return `Am înțeles: \"${p}\". Poți da mai multe detalii?`;
  }
  // For longer prompts, echo intent + a short suggestion
  const summary = p.slice(0, 140) + (p.length>140? '...' : '');
  return `Rezumat: ${summary}\n\nSugestie: Începe prin a clarifica obiectivul principal și ce constrângeri ai.`;
}

async function sendToServer(prompt){
  const sessionId = localStorage.getItem('session_id');
  try{
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({prompt, session_id: sessionId}),
    });
    if(!resp.ok) throw new Error('server error');
    const data = await resp.json();
    if(data.session_id) localStorage.setItem('session_id', data.session_id);
    return data.reply;
  }catch(err){
    return null; // signal to fallback
  }
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const value = input.value;
  if(!value) return;
  appendEntry('user', value);
  input.value = '';

  // show typing indicator
  const aiDiv = appendEntry('ai', '');
  const typingDot = document.createElement('span');
  typingDot.className = 'typing';
  aiDiv.querySelector('.text').appendChild(typingDot);
  consoleEl.scrollTop = consoleEl.scrollHeight;

  // Try server first
  const serverReply = await sendToServer(value);
  if (serverReply !== null) {
    aiDiv.querySelector('.text').innerHTML = serverReply.replace(/\n/g, '<br>');
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  if (FORCE_SERVER) {
    aiDiv.querySelector('.text').innerHTML = 'EROARE: serverul API nu răspunde. Porniți serverul cu: python chat.py --web';
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  // fallback: local simulate with a small delay
  const delay = 600 + Math.min(2000, value.length*20);
  setTimeout(()=>{
    const answer = simulateAI(value);
    aiDiv.querySelector('.text').innerHTML = answer.replace(/\n/g, '<br>');
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }, delay);
});

resetBtn.addEventListener('click', ()=>{
  consoleEl.innerHTML = '';
  appendEntry('ai', 'Consola resetată. Spune-mi cu ce pot ajuta.');
});

// initial greeting
appendEntry('ai', 'Bun venit — aceasta este o simulare de consolă. Scrie prompt-ul tău mai jos.');
input.focus();