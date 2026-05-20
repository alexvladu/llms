const consoleEl = document.getElementById('console');
const form = document.getElementById('promptForm');
const input = document.getElementById('promptInput');
const resetBtn = document.getElementById('resetBtn');
// Set to true to require server API responses (no local fallback)
const FORCE_SERVER = true;

if(window.marked){
  marked.setOptions({
    breaks: true,
    gfm: true,
  });
}

function escapeHtml(text){
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML.replace(/\n/g, '<br>');
}

function renderMarkdown(text){
  if(!window.marked || !window.DOMPurify){
    return escapeHtml(text);
  }
  return DOMPurify.sanitize(marked.parse(text || ''));
}

function setEntryText(entryEl, who, text){
  const textEl = entryEl.querySelector('.text');
  if(who === 'ai'){
    textEl.innerHTML = renderMarkdown(text);
    return;
  }
  textEl.textContent = text;
}

function appendEntry(who, text, opts = {}){
  const div = document.createElement('div');
  div.className = `entry ${who}`;
  const whoSpan = document.createElement('span');
  whoSpan.className = 'who';
  whoSpan.textContent = who === 'user' ? 'You' : 'AI';
  const textSpan = document.createElement('span');
  textSpan.className = 'text';
  div.appendChild(whoSpan);
  div.appendChild(textSpan);
  setEntryText(div, who, text);
  consoleEl.appendChild(div);
  consoleEl.scrollTop = consoleEl.scrollHeight;
  return div;
}

function simulateAI(prompt){
  const p = prompt.trim();
  if(!p) return "Nu ai scris nimic - te rog încearcă din nou.";
  if(/^(help|ajutor|ce poți|ce faci)/i.test(p)){
    return "Sunt o consolă AI simulată. Trimite orice prompt, iar eu răspund aici.";
  }
  if(p.length < 30){
    return `Am înțeles: "${p}". Poți da mai multe detalii?`;
  }
  const summary = p.slice(0, 140) + (p.length > 140 ? '...' : '');
  return `**Rezumat:** ${summary}\n\n**Sugestie:** Începe prin a clarifica obiectivul principal și ce constrângeri ai.`;
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
    return null;
  }
}

async function resetServerSession(){
  const sessionId = localStorage.getItem('session_id');
  if(!sessionId) return;

  try{
    await fetch('/api/reset', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId}),
    });
  }catch(err){
    // The visual reset should still work if the API is temporarily unavailable.
  }finally{
    localStorage.removeItem('session_id');
  }
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const value = input.value;
  if(!value) return;
  appendEntry('user', value);
  input.value = '';

  const aiDiv = appendEntry('ai', '');
  const typingDot = document.createElement('span');
  typingDot.className = 'typing';
  aiDiv.querySelector('.text').appendChild(typingDot);
  consoleEl.scrollTop = consoleEl.scrollHeight;

  const serverReply = await sendToServer(value);
  if (serverReply !== null) {
    setEntryText(aiDiv, 'ai', serverReply);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  if (FORCE_SERVER) {
    setEntryText(aiDiv, 'ai', 'EROARE: serverul API nu răspunde. Porniți serverul cu: `python chat.py --web`');
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return;
  }

  const delay = 600 + Math.min(2000, value.length * 20);
  setTimeout(()=>{
    const answer = simulateAI(value);
    setEntryText(aiDiv, 'ai', answer);
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }, delay);
});

resetBtn.addEventListener('click', async ()=>{
  await resetServerSession();
  consoleEl.innerHTML = '';
  appendEntry('ai', 'Consola resetată. Spune-mi cu ce pot ajuta.');
});

appendEntry('ai', 'Bun venit — aceasta este o simulare de consolă. Scrie prompt-ul tău mai jos.');
input.focus();